# Author: baichen318@gmail.com


import numpy as np
from collections import deque
from time import time
from dse.env.boom.env import BOOMEnv
from dse.env.rocket.env import RocketEnv
from torch.utils.tensorboard import SummaryWriter
# we do not use visdom for visualization
# from visualizer import Visualizer


class Status(object):
    """ Status """
    def __init__(self, writer):
        super(Status, self).__init__()
        self.writer = SummaryWriter(writer)
        self.perf_per_episode = deque(maxlen=10)
        self.power_per_episode = deque(maxlen=10)
        self.area_per_episode = deque(maxlen=10)
        self.step = deque(maxlen=10)
        self.episode = deque(maxlen=10)
        self.temperature = deque(maxlen=10)
        self.learning_rate = deque(maxlen=10)
        self.actor_loss_per_step = deque(maxlen=10)
        self.critic_loss_per_step = deque(maxlen=10)
        self.entropy_per_step = deque(maxlen=10)
        self.loss_per_step = deque(maxlen=10)

    def reset(self):
        self.perf_per_episode = deque(maxlen=10)
        self.power_per_episode = deque(maxlen=10)
        self.area_per_episode = deque(maxlen=10)
        self.step = deque(maxlen=10)
        self.episode = deque(maxlen=10)
        self.temperature = deque(maxlen=10)
        self.learning_rate = deque(maxlen=10)
        self.actor_loss_per_step = deque(maxlen=10)
        self.critic_loss_per_step = deque(maxlen=10)
        self.entropy_per_step = deque(maxlen=10)
        self.loss_per_step = deque(maxlen=10)

    def update_per_step(
        self,
        step,
        actor_loss,
        critic_loss,
        entropy,
        loss
    ):
        self.step.append(step)
        self.actor_loss_per_step.append(actor_loss)
        self.critic_loss_per_step.append(critic_loss)
        self.entropy_per_step.append(entropy)
        self.loss_per_step.append(loss)
        self.writer.add_scalar("actor-loss", actor_loss, step)
        self.writer.add_scalar("critic-loss", critic_loss, step)
        self.writer.add_scalar("entropy", entropy, step)
        self.writer.add_scalar("loss", loss, step)

    def update_per_episode(
        self,
        reward,
        step,
        episode,
        temperature,
        learning_rate,
        actor_loss,
        critic_loss,
        entropy,
        loss
    ):
        reward = reward.squeeze()
        self.perf_per_episode.append(reward[0])
        self.power_per_episode.append(reward[1])
        self.area_per_episode.append(reward[2])
        self.step.append(step)
        self.episode.append(episode)
        self.temperature.append(temperature)
        self.learning_rate.append(learning_rate)
        self.actor_loss_per_step.append(actor_loss)
        self.critic_loss_per_step.append(critic_loss)
        self.entropy_per_step.append(entropy)
        self.loss_per_step.append(loss)
        self.writer.add_scalar("rewards/perf", reward[0], episode)
        self.writer.add_scalar("rewards/power", reward[1], episode)
        self.writer.add_scalar("rewards/area", reward[2], episode)
        self.writer.add_scalar("rewards/mean-perf", np.mean(self.perf_per_episode), episode)
        self.writer.add_scalar("rewards/mean-power", np.mean(self.power_per_episode), episode)
        self.writer.add_scalar("rewards/mean-area", np.mean(self.area_per_episode), episode)
        self.writer.add_scalar("rewards/max-perf", np.max(self.perf_per_episode), episode)
        self.writer.add_scalar("rewards/max-power", np.max(self.power_per_episode), episode)
        self.writer.add_scalar("rewards/max-area", np.max(self.area_per_episode), episode)
        self.writer.add_scalar("rewards/min-perf", np.min(self.perf_per_episode), episode)
        self.writer.add_scalar("rewards/min-power", np.min(self.power_per_episode), episode)
        self.writer.add_scalar("rewards/min-area", np.min(self.area_per_episode), episode)
        self.writer.add_scalar("stats/temperature", temperature, episode)
        self.writer.add_scalar("stats/learning-rate", learning_rate, episode)
        self.writer.add_scalar("actor-loss", actor_loss, step)
        self.writer.add_scalar("critic-loss", critic_loss, step)
        self.writer.add_scalar("entropy", entropy, step)
        self.writer.add_scalar("loss", entropy, step)


def train_a2c(configs, agent, fixed_preference, episode, step):
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
    agent.save(episode, step)
    agent.sync_critic(episode, step)


def a2c(env, configs):
    if env == BOOMEnv:
        from dse.algo.a2c.agent import BOOMAgent as Agent
    else:
        assert env == RocketEnv
        from dse.algo.a2c.agent import RocketAgent as Agent
    agent = Agent(configs, env)

    # initialization
    step, episode = 0, 0
    state = agent.envs.reset()
    fixed_preference = agent.preference.init_preference()
    explored_preference = agent.preference.generate_preference(
        configs["num-parallel"],
        fixed_preference
    )
    status = Status(configs["summary-writer"])

    start = time()
    while episode < configs["max-episode"]:
        agent.buffer.reset()
        configs["logger"].info("[INFO]: current episode: {}, current step: {}.".format(
                episode,
                step
            )
        )
        step += configs["num-parallel"] * configs["num-step"]
        for _step in range(configs["num-step"]):
            action = agent.get_action(state, explored_preference)
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
                    info
                )
            )

            if agent.training:
                try:
                    status.update_per_step(
                        step,
                        agent.actor_loss.detach().numpy(),
                        agent.critic_loss.detach().numpy(),
                        agent.entropy.detach().numpy(),
                        agent.loss.detach().numpy()
                    )
                except AttributeError as e:
                    pass

            if done:
                configs["logger"].info(
                    "episode: {}, " \
                    "state: {}, " \
                    "next_state: {}, " \
                    "info: {}.".format(
                        episode,
                        state,
                        next_state,
                        info
                    )
                )
                state = agent.envs.reset()
                if agent.training:
                    agent.anneal()
                    try:
                        status.update_per_episode(
                            reward,
                            step,
                            episode,
                            agent.temperature,
                            agent.lr,
                            agent.actor_loss.detach().numpy(),
                            agent.critic_loss.detach().numpy(),
                            agent.entropy.detach().numpy(),
                            agent.loss.detach().numpy()
                        )
                    except AttributeError as e:
                        pass
                    for i in range(1, configs["num-parallel"]):
                        explored_preference = agent.preference.renew_preference(
                            explored_preference,
                            i
                        )
                episode += 1

            state = next_state

        if agent.training:
            train_a2c(configs, agent, fixed_preference, episode, step)

    end = time()
    configs["logger"].info(
        "done. " \
        "cost time: {}.".format(end - start)
    )
