# Author: baichen318@gmail.com


from collections import deque
from dse.env.boom.env import BOOMEnv
from dse.algo.a2c.agent import BOOMAgent
# from dse.env.rocket.env import RocketEnv
from torch.utils.tensorboard import SummaryWriter
# we do not use visdom for visualization
# from visualizer import Visualizer


class Status(object):
    """ Status """
    def __init__(self, writer):
        super(Status, self).__init__()
        self.writer = SummaryWriter(writer)
        self.perf_per_episode = deque(maxlen=100)
        self.power_per_episode = deque(maxlen=100)
        self.area_per_episode = deque(maxlen=100)
        self.step = deque(maxlen=100)
        self.episode = deque(maxlen=100)
        self.temperature = deque(maxlen=100)
        self.learning_rate = deque(maxlen=100)
        self.actor_loss_per_step = deque(maxlen=100)
        self.critic_loss_per_step = deque(maxlen=100)
        self.entropy_per_step = deque(maxlen=100)
        self.loss_per_step = deque(maxlen=100)

    def reset(self):
        self.perf_per_episode = []
        self.power_per_episode = []
        self.area_per_episode = []
        self.step = []
        self.episode = []
        self.temperature = []
        self.learning_rate = []
        self.actor_loss_per_step = []
        self.critic_loss_per_step = []
        self.entropy_per_step = []
        self.loss_per_step = []

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
        self.writer.add_scalar("perf", reward[0], episode)
        self.writer.add_scalar("power", reward[1], episode)
        self.writer.add_scalar("area", reward[2], episode)
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
        step
    )
    agent.optimize_actor_critic(total_updated_preference, discounted_reward, adv)
    agent.schedule_lr(episode)
    agent.save(episode, step)
    agent.sync_critic(episode, step)


def a2c(env, configs):
    if env == BOOMEnv:
        agent = BOOMAgent(configs, env)
    # else:
    #     assert env == RocketEnv
    #     agent = RocketAgent(configs, env)

    # initialization
    step, episode = 0, 0
    state = agent.envs.reset()
    fixed_preference = agent.preference.init_preference()
    explored_preference = agent.preference.generate_preference(
        configs["num-parallel"],
        fixed_preference
    )
    status = Status(configs["summary-writer"])

    while episode < configs["max-episode"]:
        agent.buffer.reset()
        configs["logger"].info("[INFO]: current episode: {}, current step: {}.".format(
                episode,
                step
            )
        )
        step += configs["num-parallel"] * configs["num-step"]
        for _ in range(configs["num-step"]):
            action = agent.get_action(state, explored_preference)
            next_state, reward, done, info = agent.envs.step(action)
            agent.buffer.insert(
                state,
                action,
                next_state,
                reward,
                done
            )

            # a tricky part to output the best state
            if done:
                configs["logger"].info(
                    "episode: {}, " \
                    "state: {}, " \
                    "next_state: {}, " \
                    "reward: {}.".format(
                        episode,
                        state,
                        next_state,
                        info
                    )
                )

            state = next_state

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

        if agent.training:
            train_a2c(configs, agent, fixed_preference, episode, step)
