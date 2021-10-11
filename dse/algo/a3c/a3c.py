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
            values = values.gather(
                1, actions.view(-1, action_shape).view(-1, 1, 1).expand(values.size(0), 1, values.size(2))
            ).view(-1, reward_shape)
            return values, torch.bmm(
                torch.autograd.Variable(w.repeat(self.configs["num-process"], 1).unsqueeze(1)),
                values.unsqueeze(2)
            ).squeeze()

        num_steps, num_processes, _ = buffer.rewards.size()

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            buffer.obs[:-1].view(-1, *obs_shape),
            w.repeat(self.configs["num-process"], 1),
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
    episode_rewards = deque(maxlen=10)
    total_rewards, ipc, power, area = [], [], [], []
    start = time.time()
    num_updates = configs["num-train-step"] // configs["n-step-td"] // configs["num-process"]
    for i in range(num_updates):
        # Omega preference space on PPA metrics
        w = torch.randn(configs["num-process"], len(configs["metrics"]))
        w = torch.abs(w) / torch.sum(torch.abs(w))
        for step in range(configs["n-step-td"]):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(buffer.obs[step], w)
            obs, reward, done, info = envs.step(action)
            msg = "[INFO]: action: {}, obs: {}, reward: {}, done: {}".format(
                action, obs, reward, done
            )
            configs["logger"].info(msg)
            for r in reward:
                # NOTICE: refers to dse/env/rocket/design_env.py
                assert len(r) == len(configs["metrics"]), "[ERROR]: metrics are unsupported."
                ipc.append(r[0])
                # unit: w
                power.append((-r[1] / 10) * 1e3)
                # unit: mm^2
                area.append(-r[2])
                r = r * (1 / 3)
                episode_rewards.append(
                    torch.sum(r)
                )
                total_rewards.append(
                    torch.sum(r)
                )

            masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in done]
            )
            buffer.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            _, next_value = actor_critic.get_value(
                buffer.obs[-1],
                w
            )
            next_value = next_value.detach()
        buffer.compute_returns(next_value, configs["gamma"])
        value_loss, action_loss, dist_entropy = agent.update(buffer, w)
        agent.homotopy_optimization()
        buffer.after_update()
        if i % configs["save-interval"] == 0 and len(episode_rewards) > 1:
            total_num_steps = (i + 1) * configs["num-process"] * configs["n-step-td"]
            end = time.time()
            msg = "[INFO]: updates {}, num timesteps {}, FPS {:.4f}. Last {} training episode. ".format(
                i,
                total_num_steps,
                total_num_steps / (end - start),
                len(episode_rewards)
            )
            msg += "mean/median reward: {:.4f}/{:.4f}, min/max reward: {:.4f}/{:.4f}. ".format(
                np.mean(episode_rewards),
                np.median(episode_rewards),
                np.min(episode_rewards),
                np.max(episode_rewards)
            )
            msg += "value loss: {:.4f}, action loss: {:.4f}, entroy loss: {:.4f}. ".format(
                value_loss,
                action_loss,
                dist_entropy
            )
            msg += "max reward util now: {:.4f}.".format(np.max(total_rewards))
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
                OrderedDict({
                        "value loss": value_loss,
                        "action loss": action_loss,
                        "entropy loss": dist_entropy
                    }
                ),
                OrderedDict({
                        "mean episode reward": np.mean(episode_rewards),
                        "median episode reward": np.median(episode_rewards),
                        "min. episode reward": np.min(episode_rewards),
                        "max. episode reward": np.max(episode_rewards)
                    }
                ),
                OrderedDict({
                        "max. reward": np.max(total_rewards)
                    }
                ),
                OrderedDict({
                        "mean reward": np.mean(total_rewards),
                        "median reward": np.median(total_rewards),
                        "min. reward": np.min(total_rewards),
                        "max. reward": np.max(total_rewards),
                    }
                ),
                OrderedDict({
                        "mean IPC": np.mean(ipc),
                        "median IPC": np.median(ipc),
                        "min. IPC": np.min(ipc),
                        "max. IPC": np.max(ipc),
                        "mean power": np.mean(power),
                        "median power": np.median(power),
                        "min. power": np.min(power),
                        "max. power": np.max(power),
                        "mean area": np.mean(area),
                        "median area": np.median(area),
                        "min. area": np.min(area),
                        "max. area": np.max(area)
                    }
                )
            )
