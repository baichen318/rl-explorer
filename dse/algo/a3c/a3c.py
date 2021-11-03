# Author: baichen318@gmail.com

import time
import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque, OrderedDict
from dse.algo.a3c.util import make_vec_envs
from dse.algo.a3c.buffer import Buffer
from dse.algo.a3c.model import Policy
from visualizer import Visualizer


class A3CAgent():
    def __init__(
        self,
        configs,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        lr,
        eps,
        alpha,
        max_grad_norm,
    ):
        self.configs = configs
        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.beta = configs["beta"]
        self.beta_init = self.beta
        self.uplim = 1.00
        self.tau = 1000.
        # scale `self.expbase` manually to enlarge the base
        self.expbase = float(np.power(
                self.tau * (self.uplim - self.beta),
                # 1. / (self.configs["num-train-step"] * self.configs["n-step-td"] * self.configs["num-env-step"] * self.configs["num-process"])
                1. / self.configs["num-train-step"]
            )
        )
        self._delta = self.expbase / self.tau
        self.optimizer = torch.optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha
        )

    def update(self, buffer, w):
        obs_shape = buffer.obs.size()[2:]
        action_shape = buffer.actions.size()[-1]
        reward_shape = buffer.rewards.size()[-1]

        def scalarize(values, actions):
            # NOTICE: Key point in MOA3C, we need to optimize along the
            # current best direction w.r.t. Omega space, simultaneously,
            # we need to return values for multi-objective optimizations
            # values: <n-step-td x num-process, 3>
            # w_values: <n-step-td x num-process, 1>
            values = values.gather(
                1, actions.view(-1, action_shape).view(-1, 1, 1).expand(values.size(0), 1, values.size(2))
            ).view(-1, reward_shape)
            return values, torch.bmm(
                torch.autograd.Variable(w.view(-1, reward_shape).unsqueeze(1)),
                values.unsqueeze(2)
            ).squeeze()

        num_steps, num_processes, _ = buffer.rewards.size()

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            buffer.obs[:-1].view(-1, *obs_shape),
            w.view(-1, reward_shape),
            buffer.actions.view(-1, action_shape)
        )

        # values = values.view(num_steps, num_processes, 1)
        # NOTICE: `action_shape` refers to one single action while `action_space.n` or
        # `buffer.value_preds.size(-2)` refers to the dimension of the action space
        # values = values.view(num_steps, num_processes, buffer.value_preds.size(-2), reward_shape)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        values, w_values = scalarize(values, buffer.actions)
        returns, w_returns = scalarize(
            buffer.returns[:-1].view(-1, self.actor_critic.action_size, reward_shape),
            buffer.actions
        )

        # NOTICE: beta * || w^TR - w^TV ||^2 + (1 - beta) * || R - V ||^2
        # NOTICE: advantages = w^TR - w^TV
        value_loss = self.beta * (w_returns - w_values).pow(2).mean() + \
            (1 - self.beta) * (returns.view(-1) - values.view(-1)).pow(2).mean()

        advantages = w_returns - w_values
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - \
            dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def homotopy_optimization(self):
        self.beta += self._delta
        self._delta = (self.beta - self.beta_init) * self.expbase + self.beta_init - self.beta

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        self.configs["logger"].info("[INFO]: saving agent to %s" % path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.configs["logger"].info("[INFO]: loading agent from %s" % path)


def a3c(env, configs):
    device = torch.device("cuda:0" if configs["cuda"] else "cpu")
    envs = make_vec_envs(env, configs, device)

    actor_critic = Policy(
        configs,
        envs.observation_space.shape,
        envs.action_space
    )
    actor_critic.to(device)

    agent = A3CAgent(
        configs,
        actor_critic,
        value_loss_coef=configs["value-loss-weight"],
        entropy_coef=configs["entropy-weight"],
        lr=configs["lr"],
        eps=configs["eps"],
        alpha=configs["alpha"],
        max_grad_norm=configs["max-grad-norm"]
    )

    buffer = Buffer(
        configs,
        envs.observation_space.shape,
        envs.action_space
    )

    visualizer = Visualizer(configs)
    # reset
    obs = envs.reset()
    configs["logger"].info("[INFO]: initialized status: {}".format(obs))
    buffer.obs[0].copy_(obs)
    buffer.to(device)
    episode_ppa = {
        "ipc": deque(maxlen=10),
        "power": deque(maxlen=10),
        "area": deque(maxlen=10)
    }
    ppa = {
        "ipc": [],
        "power": [],
        "area": []
    }
    start = time.time()
    num_updates = configs["num-train-step"] // configs["n-step-td"] // configs["num-process"]
    for i in range(num_updates):
        # NOTICE: Omega preference space on PPA metrics
        w = torch.randn(configs["num-process"], len(configs["metrics"]))
        w = torch.abs(w) / torch.sum(torch.abs(w))
        w = w.view(-1, configs["num-process"], len(configs["metrics"])).repeat(configs["n-step-td"], 1, 1)
        msg = "[INFO]: preference vector: {}".format(w)
        configs["logger"].info(msg)
        for step in range(configs["n-step-td"]):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(buffer.obs[step], w[step])
            obs, reward, done, info = envs.step(action)
            msg = "[INFO]: action: {}, obs: {}, reward: {}, done: {}".format(
                action, obs, reward, done
            )
            configs["logger"].info(msg)
            for r in reward:
                assert len(r) == len(configs["metrics"]), "[ERROR]: metrics are unsupported."
                episode_ppa["ipc"].append(r[0])
                ppa["ipc"].append(r[0])
                episode_ppa["power"].append(-r[1])
                ppa["power"].append(-r[1])
                episode_ppa["area"].append(-r[2])
                ppa["area"].append(-r[2])

            masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in done]
            )
            buffer.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            _, next_value = actor_critic.get_value(
                buffer.obs[-1],
                # w is the same in every TD step
                w[-1]
            )
            next_value = next_value.detach()
        buffer.compute_returns(next_value, configs["gamma"])
        value_loss, action_loss, dist_entropy = agent.update(buffer, w)
        agent.homotopy_optimization()
        buffer.after_update()
        # NOTICE: we display VALID rewards
        _ipc = np.array(sorted(episode_ppa["ipc"]))
        j = _ipc[np.where(_ipc != -1)]
        if j.shape[0] == 0:
            j = -1
        else:
            j = j[0]
        if i % configs["save-interval"] == 0:
            total_num_steps = (i + 1) * configs["num-process"] * configs["n-step-td"]
            end = time.time()
            msg = "[INFO]: updates {}, num timesteps {}, FPS {:.4f}. Last {} training episode. ".format(
                i,
                total_num_steps,
                total_num_steps / (end - start),
                len(episode_ppa["ipc"])
            )
            msg += "value loss: {:.4f}, action loss: {:.4f}, entroy loss: {:.4f}. ".format(
                value_loss,
                action_loss,
                dist_entropy
            )
            configs["logger"].info(msg)
            agent.save(
                os.path.join(
                    configs["model-path"],
                    "a3c-%d.pt" % (i + 1)
                )
            )
            visualizer.plot_current_status(
                i,
                i / num_updates,
                "loss",
                OrderedDict({
                        "value loss": value_loss,
                        "action loss": action_loss,
                        "entropy loss": dist_entropy
                    }
                ),
                title="MOA3C Losses Over Time",
                xlabel="Epoch",
                ylabel="Loss",
                win=1
            )
            visualizer.plot_current_status(
                i,
                i / num_updates,
                "episode_ppa",
                OrderedDict({
                        "mean episode IPC": np.mean(episode_ppa["ipc"]),
                        "median episode IPC": np.median(episode_ppa["ipc"]),
                        "min. episode IPC": j,
                        "max. episode IPC": np.max(episode_ppa["ipc"]),
                        "mean episode power": np.mean(episode_ppa["power"]),
                        "median episode power": np.median(episode_ppa["power"]),
                        "min. episode power": np.min(episode_ppa["power"]),
                        "max. episode power": np.max(episode_ppa["power"]),
                        "mean episode area": np.mean(episode_ppa["area"]),
                        "median episode area": np.median(episode_ppa["area"]),
                        "min. episode area": np.min(episode_ppa["area"]),
                        "max. episode area": np.max(episode_ppa["area"]),
                    }
                ),
                title="Episode PPA Over Time",
                xlabel="Epoch",
                ylabel="Episode PPA",
                win=2
            )
            visualizer.plot_current_status(
                i,
                i / num_updates,
                "ppa",
                OrderedDict({
                        "mean IPC": np.mean(ppa["ipc"]),
                        "median IPC": np.median(ppa["ipc"]),
                        "min. IPC": j,
                        "max. IPC": np.max(ppa["ipc"]),
                        "mean power": np.mean(ppa["power"]),
                        "median power": np.median(ppa["power"]),
                        "min. power": np.min(ppa["power"]),
                        "max. power": np.max(ppa["power"]),
                        "mean area": np.mean(ppa["area"]),
                        "median area": np.median(ppa["area"]),
                        "min. area": np.min(ppa["area"]),
                        "max. area": np.max(ppa["area"])
                    }
                ),
                title="PPA Over Time",
                xlabel="Epoch",
                ylabel="PPA",
                win=3
            )

def evaluate_a3c(env, configs):
    device = torch.device("cuda:0" if configs["cuda"] else "cpu")
    envs = make_vec_envs(env, configs, device)
    actor_critic = Policy(
        configs,
        envs.observation_space.shape,
        envs.action_space
    )

    preference = torch.Tensor(configs["preference"]).unsqueeze(0)
    preference = torch.abs(preference) / torch.sum(torch.abs(preference))

    if configs["mode"] == "evaluate":
        model_list = os.listdir(os.path.join(configs["model"]))
        model_list.sort(key=lambda x: int(x[4:].strip(".pt")))
        # NOTICE: we evaluate models with 15% - 75%
        start = int(len(model_list) * 0.75)
        end = int(len(model_list) * 0.95)
        models = model_list[start: end]
    else:
        assert configs["mode"] == "explore", \
            "[ERROR]: %s is unsupported." % configs["mode"]
        models = [configs["model"]]
    for model in models:
        if configs["mode"] == "evaluate":
            if model.endswith(".pt"):
                model = os.path.join(configs["model"], model)
            else:
                continue
        ppa = {
            "ipc": [],
            "power": [],
            "area": []
        }
        actor_critic.load(model)
        for i in range(1, 10 + 1):
            obs = envs.reset()
            s = time.time()
            while len(ppa["ipc"]) < 30:
                with torch.no_grad():
                    _, action, _ = actor_critic.act(
                        obs,
                        preference
                    )
                    obs, reward, done, info = envs.step(action)

                    for r in reward:
                        assert len(r) == len(configs["metrics"]), "[ERROR]: metrics are unsupported."
                        ppa["ipc"].append(r[0])
                        ppa["power"].append(-r[1])
                        ppa["area"].append(-r[2])
            e = time.time()
            msg = "[INFO]: round: {}, time: {}, evaluate using {}, {} episodes: mean IPC: {:.4f}, mean Power: {:.4f}, mean Area: {:.4f}\n".format(
                i, e - s, model, len(ppa["ipc"]), np.mean(ppa["ipc"]), np.mean(ppa["power"]), np.mean(ppa["area"])
            )
            configs["logger"].info(msg)
