# Author: baichen318@gmail.com


import numpy as np
from time import time
from dse.env.boom.env import BOOMEnv
from dse.env.rocket.env import RocketEnv
from collections import deque, OrderedDict
from utils.utils import Timer, assert_error
from torch.utils.tensorboard import SummaryWriter
from dse.algo.a2c.functions import tensor_to_array


class Status(object):
    """
        Status
    """

    class BOOMStatus(object):
        def __init__(self, action_prob):
            """
                IFU action probability
                maxBrCount action probability
                ROB action probability
                PRF action probability
                ISU action probability
                LSU action probability
                D-Cache action probability
            """
            self.isu_action_prob = action_prob[0]
            self.ifu_action_prob = action_prob[1]
            self.maxBrCount_action_prob = action_prob[2]
            self.rob_action_prob = action_prob[3]
            self.prf_action_prob = action_prob[4]
            self.lsu_action_prob = action_prob[5]
            self.bp_action_prob = action_prob[6]
            self.icache_action_prob = action_prob[7]
            self.dcache_action_prob = action_prob[8]

        def update_bp_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.bp_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/BP",
                action_prob,
                episode
            )

        def update_ifu_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.ifu_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/IFU",
                action_prob,
                episode
            )

        def update_maxBrCount_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.maxBrCount_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/maxBrCount",
                action_prob,
                episode
            )

        def update_rob_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.rob_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/ROB",
                action_prob,
                episode
            )

        def update_prf_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.prf_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/PRF",
                action_prob,
                episode
            )

        def update_isu_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.isu_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/ISU",
                action_prob,
                episode
            )

        def update_lsu_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.lsu_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/LSU",
                action_prob,
                episode
            )

        def update_icache_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.icache_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/I$",
                action_prob,
                episode
            )

        def update_dcache_action_prob(self, writer, episode):
            action_prob = OrderedDict()
            idx = 1
            for prob in np.average(self.dcache_action_prob, axis=0):
                action_prob['a' + str(idx)] = prob
                idx += 1
            writer.add_scalars(
                "episode/action/D$",
                action_prob,
                episode
            )

        def update(self, writer, episode):
            self.update_isu_action_prob(writer, episode)
            self.update_ifu_action_prob(writer, episode)
            self.update_maxBrCount_action_prob(writer, episode)
            self.update_rob_action_prob(writer, episode)
            self.update_prf_action_prob(writer, episode)
            self.update_lsu_action_prob(writer, episode)
            self.update_bp_action_prob(writer, episode)
            self.update_icache_action_prob(writer, episode)
            self.update_dcache_action_prob(writer, episode)


    def __init__(self, configs):
        super(Status, self).__init__()
        self.configs = configs
        self.writer = SummaryWriter(configs["summary-writer"])

    @property
    def design(self):
        return self.configs["algo"]["design"]

    def update_perf(self, perf, episode):
        self.writer.add_scalars(
            "episode/perf",
            {
                "perf": perf,
            },
            episode
        )

    def update_power(self, power, episode):
        self.writer.add_scalars(
            "episode/power",
            {
                "power": power,
            },
            episode
        )

    def update_area(self, area, episode):
        self.writer.add_scalars(
            "episode/area",
            {
                "area": area,
            },
            episode
        )

    def update_reward(self, reward, episode):
        self.writer.add_scalars(
            "episode/reward",
            {
                "reward": reward,
            },
            episode
        )

    def update_preference(self, w, episode):
        perf_w, power_w, area_w = np.average(w, axis=0)
        self.writer.add_scalars(
            "episode/preference",
            {
                "perf-preference": perf_w,
                "power-preference": power_w,
                "area-performance": area_w,
            },
            episode
        )

    def update_loss(self, agent, episode):
        self.writer.add_scalars(
            "episode/loss",
            {
                "actor-loss": agent.actor_loss.detach().numpy(),
                "critic-loss": agent.critic_loss.detach().numpy(),
                "entropy": agent.entropy.detach().numpy(),
                "total-loss": agent.loss.detach().numpy()
            },
            episode
        )

    def update_learning_rate(self, agent, episode):
        self.writer.add_scalars(
            "episode/learning-rate",
            {
                "learning-rate": agent.lr
            },
            episode
        )

    def update_action_prob(self, action_prob, episode):
        if "BOOM" in self.design:
            self.BOOMStatus(action_prob).update(self.writer, episode)

    def update(self, agent, explore_w, action_prob, episode):
        """
            Record several information w.r.t. the episode, including
            1. performance
            2. power
            3. area
            4. PPA values w.r.t. the uniform preference
            5. average performance preference
            6. average power preference
            7. average area preference
            8. actor loss
            9. critic loss
            10. entropy loss
            11. learning rate
            12. design-related action probabilities
        """
        buffer = agent.buffer
        perf, power, area = np.average(buffer.total_reward[-1], axis=0)
        self.update_perf(perf, episode)
        self.update_power(power, episode)
        self.update_area(area, episode)
        reward = abs(perf) + abs(power) + abs(area)
        self.update_reward(reward, episode)
        self.update_preference(explore_w, episode)
        self.update_loss(agent, episode)
        self.update_learning_rate(agent, episode)
        self.update_action_prob(action_prob, episode)


def train_ppo_impl(configs, agent, fixed_w, episode, status):
    updated_w = agent.preference.generate_preference(
        agent.sample_size,
        fixed_w
    )
    total_updated_w = updated_w.repeat(
        agent.num_parallel * \
            agent.num_step,
        axis=0
    )
    agent.buffer.generate_batch_with_n_step()
    value, next_value, policy = agent.forward_transition(total_updated_w)
    discounted_reward = agent.calc_discounted_reward(value, next_value)
    adv, discounted_reward = agent.envelope_operator(
        updated_w,
        discounted_reward,
        value,
        episode
    )
    agent.optimize_actor_critic(total_updated_w, discounted_reward, adv)
    agent.schedule_lr(episode)
    agent.save(episode)
    agent.sync_critic(episode)


def train_ppo(agent, configs):
    envs = agent.envs
    assert agent.num_step == \
        envs.safe_get_attr("dims_of_tunable_state"), \
            assert_error(": num_step {} vs " \
                " tunabe state: {}".format(
                    agent.num_step,
                    envs.safe_get_attr("dims_of_tunable_state")
                )
            )

    preference = agent.preference

    # initialization
    fixed_w = preference.init_preference()
    explore_w = preference.generate_preference(
        agent.num_parallel,
        fixed_w
    )
    status = Status(configs)

    with Timer("{}".format(agent.mode)):
        for episode in range(agent.max_episode):
            """
                When we start an episode, we clear the buffer.
            """
            agent.buffer.reset()
            state = envs.reset()

            action_prob = []
            old_explore_w = explore_w

            for _ in range(agent.num_step):
                action, policy = agent.get_action(state, explore_w)
                next_state, reward, done, info = envs.step(action)

                """
                    If `done` is True, `next_state` is override,
                    so we need to get the correct `next_state` from
                    `info`.
                    See: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/subproc_vec_env.py#L22
                """
                if done[0]:
                    # stack all terminal observation(s)
                    _next_state = np.stack(
                        [_info["terminal_observation"] for _info in info]
                    )
                    agent.buffer.insert(
                        state, action, _next_state, reward, done
                    )
                    configs["logger"].info(
                        "state: {}, action: {}, next_state: {}, reward: {}".format(
                            state, action, _next_state, reward
                        )
                    )
                else:
                    agent.buffer.insert(
                        state, action, next_state, reward, done
                    )
                    configs["logger"].info(
                        "state: {}, action: {}, next_state: {}, reward: {}".format(
                            state, action, next_state, reward
                        )
                    )

                # for visualization
                action_prob.append(policy)

                for i in range(1, agent.num_parallel):
                    if done[i]:
                        # if the i-th agent finishes, then we renew the preference
                        explore_w = preference.renew_preference(
                            explore_w, i
                        )
                state = next_state

            agent.anneal()
            train_ppo_impl(configs, agent, fixed_w, episode, status)
            status.update(agent, old_explore_w, action_prob, episode)


def test_ppo():
    pass


def ppo(env, configs):
    if env == BOOMEnv:
        from dse.algo.ppo.agent import BOOMAgent as Agent
    else:
        assert env == RocketEnv
        from dse.algo.ppo.agent import RocketAgent as Agent
    agent = Agent(configs, env)

    if configs["algo"]["mode"] == "train":
        train_ppo(agent, configs)
    else:
        test_ppo()
