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

        self.optimizer = torch.optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha
        )

    def update(self, buffer):
        obs_shape = buffer.obs.size()[2:]
        action_shape = buffer.actions.size()[-1]
        num_steps, num_processes, _ = buffer.rewards.size()

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            buffer.obs[:-1].view(-1, *obs_shape),
            buffer.masks[:-1].view(-1, 1),
            buffer.actions.view(-1, action_shape)
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = buffer.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - \
            dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

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
        envs.observation_space.shape,
        envs.action_space,
        configs["hidden-size"]
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
        configs["n-step-td"],
        configs["num-process"],
        envs.observation_space.shape,
    )

    visualizer = Visualizer(configs)
    # reset
    obs = envs.reset()
    configs["logger"].info("[INFO]: initialized status: {}".format(obs))
    buffer.obs[0].copy_(obs)
    buffer.to(device)
    episode_rewards = deque(maxlen=10)
    total_rewards = []
    start = time.time()
    num_updates = configs["num-env-step"] // configs["n-step-td"] // configs["num-process"]
    for i in range(num_updates):
        for step in range(configs["n-step-td"]):
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    buffer.obs[step],
                    buffer.masks[step]
                )
            obs, reward, done, info = envs.step(action)
            msg = "[INFO]: action: {}, obs: {}, reward: {}, done: {}, info: {}".format(
                action, obs, reward, done, info
            )
            configs["logger"].info(msg)

            for _info in info:
                if "reward" in _info.keys():
                    episode_rewards.append(_info["reward"])
                    total_rewards.append(_info["reward"])

            masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in done]
            )
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in _info.keys() else [1.0]
                 for _info in info]
            )
            buffer.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                buffer.obs[-1],
                buffer.masks[-1]
            ).detach()
        buffer.compute_returns(next_value, configs["gamma"])
        value_loss, action_loss, dist_entropy = agent.update(buffer)
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
            msg += "mean/median reward {:.4f}/{:.4f}, min/max reward {:.4f}/{:.4f}. ".format(
                np.mean(episode_rewards),
                np.median(episode_rewards),
                np.min(episode_rewards),
                np.max(episode_rewards)
            )
            msg += "value loss: {:.4f}, action loss: {:.4f}, entroy loss: {:.4f}".format(
                value_loss,
                action_loss,
                dist_entropy
            )
            configs["logger"].info(msg)
            agent.save(
                os.path.join(
                    configs["model-path"],
                    "a3c.pt"
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
                        "mean total reward": np.mean(total_rewards),
                        "median total reward": np.median(total_rewards),
                        "min. total reward": np.min(total_rewards),
                        "max. total reward": np.max(total_rewards)
                    }
                )
            )
