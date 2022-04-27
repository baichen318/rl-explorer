# Author: baichen318@gmail.com


from dse.env.boom.env import BOOMEnv
from dse.algo.a2c.agent import BOOMAgent
# from dse.env.rocket.env import RocketEnv


def train_a2c(configs, agent, fixed_preference, step):
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
    agent.schedule_lr(step)
    agent.save(step)
    agent.sync_critic(step)


def a2c(env, configs):
    if env == BOOMEnv:
        agent = BOOMAgent(configs, env)
    # else:
    #     assert env == RocketEnv
    #     agent = RocketAgent(configs, env)

    step = 0
    state = agent.envs.reset()
    fixed_preference = agent.preference.init_preference()
    explored_preference = agent.preference.generate_preference(
        configs["num-parallel"],
        fixed_preference
    )

    while step < configs["max-episode"]:
        agent.buffer.reset()
        configs["logger"].info("[INFO]: current step: {}.".format(step))
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
            state = next_state

            # TODO
            if done:
                pass

        train_a2c(configs, agent, fixed_preference, step)
