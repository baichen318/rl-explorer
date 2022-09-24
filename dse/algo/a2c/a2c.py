# Author: baichen318@gmail.com


import numpy as np
from time import time
from collections import deque, OrderedDict
from dse.env.boom.env import BOOMEnv
from dse.env.rocket.env import RocketEnv
from torch.utils.tensorboard import SummaryWriter
from dse.algo.a2c.functions import tensor_to_array


class Status(object):
    """ Status """
    def __init__(self, writer):
        super(Status, self).__init__()
        self.writer = SummaryWriter(writer)
        self.reset()

    def reset(self):
        self.perf_pred = []
        self.perf_preference = []
        self.power_pred = []
        self.power_preference = []
        self.area_pred = []
        self.area_preference = []
        self.actor_loss = []
        self.critic_loss = []
        self.entropy = []
        self.loss = []
        self.action = []

    def update_per_episode(self, agent, info, preference, episode):
        info = info[0]
        self.perf_pred.append(float(info["perf-pred"]))
        self.perf_preference.append(float(preference[0][0]))
        self.power_pred.append(float(info["power-pred"]))
        self.power_preference.append(float(preference[0][1]))
        self.area_pred.append(float(info["area-pred"]))
        self.area_preference.append(float(preference[0][2]))
        self.writer.add_scalars(
            "episode/perf",
            {
                "perf-baseline": float(info["perf-baseline"]),
                "perf-pred": float(info["perf-pred"]),
                "perf-preference": float(preference[0][0])
            },
            episode
        )
        self.writer.add_scalars(
            "episode/power",
            {
                "power-baseline": float(info["power-baseline"]),
                "power-pred": float(info["power-pred"]),
                "power-preference": float(preference[0][1])
            },
            episode
        )
        self.writer.add_scalars(
            "episode/area",
            {
                "area-baseline": float(info["area-baseline"]),
                "area-pred": float(info["area-pred"]),
                "area-preference": float(preference[0][2])
            },
            episode
        )
        self.writer.add_scalars(
            "episode/preference",
            {
                "perf-preference": float(preference[0][0]),
                "power-preference": float(preference[0][1]),
                "area-preference": float(preference[0][2])
            },
            episode
        )
        if hasattr(agent, "actor_loss") and \
            hasattr(agent, "critic_loss") and \
            hasattr(agent, "entropy") and \
            hasattr(agent, "loss"):
            self.actor_loss.append(agent.actor_loss.detach().numpy())
            self.critic_loss.append(agent.critic_loss.detach().numpy())
            self.entropy.append(agent.entropy.detach().numpy())
            self.loss.append(agent.loss.detach().numpy())
            self.writer.add_scalars(
                "episode/loss",
                {
                    "actor-loss": agent.actor_loss.detach().numpy(),
                    "critic-loss": agent.critic_loss.detach().numpy(),
                    "entropy": agent.entropy.detach().numpy(),
                    "loss": agent.loss.detach().numpy()
                },
                episode
            )

    def update_action_per_episode(self, action, episode):
        # num-process = 1
        action = tensor_to_array(action)
        self.action.append(action[0])
        action_dict = OrderedDict()
        cnt = 1
        for a in action[0]:
            action_dict['a' + str(cnt)] = a
            cnt += 1
        self.writer.add_scalars(
            "episode/action",
            action_dict,
            episode
        )


    def update_per_sequence(self, agent, info, preference, sequence):
        info = info[0]
        self.perf_pred.append(float(info["perf-pred"]))
        self.perf_preference.append(float(preference[0][0]))
        self.power_pred.append(float(info["power-pred"]))
        self.power_preference.append(float(preference[0][1]))
        self.area_pred.append(float(info["area-pred"]))
        self.area_preference.append(float(preference[0][2]))
        self.writer.add_scalars(
            "sequence/perf",
            {
                "mean-perf": np.mean(self.perf_pred),
                "max-perf": np.max(self.perf_pred),
                "min-perf": np.min(self.perf_pred)
            },
            sequence
        )
        self.writer.add_scalars(
            "sequence/power",
            {
                "mean-power": np.mean(self.power_pred),
                "max-power": np.max(self.power_pred),
                "min-power": np.min(self.power_pred)
            },
            sequence
        )
        self.writer.add_scalars(
            "sequence/area",
            {
                "mean-area": np.mean(self.area_pred),
                "max-area": np.max(self.area_pred),
                "min-area": np.min(self.area_pred)
            },
            sequence
        )
        self.writer.add_scalars(
            "sequence/preference",
            {
                "perf-preference": np.mean(self.perf_preference),
                "power-preference": np.mean(self.power_preference),
                "area-preference": np.mean(self.area_preference)
            },
            sequence
        )
        self.writer.add_scalars(
            "sequence/loss",
            {
                "actor-loss": np.mean(self.actor_loss),
                "critic-loss": np.mean(self.critic_loss),
                "entropy": np.mean(self.entropy),
                "loss": np.mean(self.loss)
            },
            sequence
        )
        action = np.array(self.action).mean(axis=0)
        action_dict = OrderedDict()
        cnt = 1
        for a in action:
            action_dict['a' + str(cnt)] = a
            cnt += 1
        self.writer.add_scalars(
            "sequence/action",
            action_dict,
            sequence
        )
        self.reset()


def status_update_per_episode(status, agent, info, preference, episode):
    if agent.training:
        status.update_per_episode(agent, info, preference, episode)


def status_update_per_sequence(status, agent, info, preference, sequence):
    status.update_per_sequence(agent, info, preference, sequence)


def train_a2c(configs, status, agent, fixed_preference, episode):
    updated_preference = agent.preference.generate_preference(
        configs["sample-size"],
        fixed_preference
    )
    total_updated_preference = updated_preference.repeat(
        configs["num-parallel"] * configs["num-step"],
        axis=0
    )
    agent.buffer.generate_batch_with_n_step()
    value, next_value, _ = agent.forward_transition(total_updated_preference)
    discounted_reward = agent.calc_discounted_reward(value, next_value)
    adv, discounted_reward = agent.calc_advantage(
        updated_preference,
        discounted_reward,
        value,
        episode
    )
    agent.optimize_actor_critic(total_updated_preference, discounted_reward, adv)
    agent.schedule_lr(episode)
    agent.save(episode)
    agent.sync_critic(episode)


def a2c(env, configs):
    if env == BOOMEnv:
        from dse.algo.a2c.agent import BOOMAgent as Agent
    else:
        assert env == RocketEnv
        from dse.algo.a2c.agent import RocketAgent as Agent
    agent = Agent(configs, env)

    # initialization
    iteration, episode, sequence = 0, 0, 0
    state = agent.envs.reset()
    fixed_preference = agent.preference.init_preference()
    explored_preference = agent.preference.generate_preference(
        configs["num-parallel"],
        fixed_preference
    )
    status = Status(configs["summary-writer"])

    start = time()
    while iteration < configs["max-iteration"]:
        agent.buffer.reset()
        configs["logger"].info(
            "[INFO]: current iteration: {}, current episode {}.".format(
                iteration,
                episode
            )
        )
        for _step in range(configs["num-step"]):
            action = agent.get_action(state, explored_preference, status, episode)
            next_state, reward, done, info = agent.envs.step(action)
            agent.buffer.insert(
                state,
                action,
                next_state,
                reward,
                done
            )

            configs["logger"].info(
                "episode: {}, " \
                "step: {}, " \
                "state: {}, " \
                "action: {}, " \
                "next_state: {}, " \
                "info: {}.".format(
                    episode,
                    _step + 1,
                    state,
                    action,
                    next_state,
                    info[0]
                )
            )

            state = next_state

            status_update_per_episode(
                status, agent, info, explored_preference, episode
            )

            if done:
                state = agent.envs.reset()
                agent.anneal()
                status_update_per_sequence(
                    status, agent, info, explored_preference, sequence
                )
                sequence += 1
                # cannot reach
                for i in range(1, configs["num-parallel"]):
                    explored_preference = agent.preference.renew_preference(
                        explored_preference,
                        i
                    )

            episode += configs["num-parallel"]

        if agent.training:
            train_a2c(configs, status, agent, fixed_preference, episode)

    end = time()
    configs["logger"].info(
        "done. " \
        "cost time: {}.".format(end - start)
    )
