import gym
from gym.spaces import Box, Discrete
import numpy as np

from pendulum_ppo_tensorflow_ver3 import ActorCritic, PPOObserver, ReplayBuffer, PPOMaster

def run(env, actor_critic, replay_buffer):
    observation = env.reset()
    done = False
    action, v, logp = actor_critic.step(observation)
    if isinstance(env.action_space, Discrete):
        _, reward, done, _ = env.step(action[0])
    elif isinstance(env.action_space, Box):
        _, reward, done, _ = env.step(action)
    for i in range(replay_buffer.max_size):
        replay_buffer.store(observation, action, reward, v, logp)
    replay_buffer.finish_path()
    data = replay_buffer.get()
    return data

def test_observer():
    env = gym.make("CartPole-v0")
    env = PPOObserver(env)
    assert env.reset() is not None
    assert env.observation_space is not None
    assert env.observation_space.shape[0] == 4
    assert env.action_space is not None
    assert env.action_space.n == 2
    assert env.render() is not None
    assert env.step(env.action_space.sample()) is not None

    env = gym.make("Pendulum-v0")
    env = PPOObserver(env)
    assert env.reset() is not None
    assert env.observation_space is not None
    assert env.observation_space.shape[0] == 3
    assert env.action_space is not None
    assert env.action_space.shape[0] == 1
    assert env.render() is not None
    assert env.step(env.action_space.sample()) is not None

def test_actor_critic():
    env = gym.make("CartPole-v0")
    env = PPOObserver(env)
    observation = env.reset()
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    assert actor_critic.step(observation) is not None
    assert actor_critic.act(observation) is not None
    env.close()

    env = gym.make("Pendulum-v0")
    env = PPOObserver(env)
    observation = env.reset()
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    assert actor_critic.step(observation) is not None
    assert actor_critic.act(observation) is not None
    env.close()

def test_replay_buffer():
    env = gym.make("CartPole-v0")
    env = PPOObserver(env)
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape, 128)
    run(env, actor_critic, replay_buffer)

    env = gym.make("Pendulum-v0")
    env = PPOObserver(env)
    actor_critic = ActorCritic(env.observation_space, env.action_space)
    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape, 128)
    run(env, actor_critic, replay_buffer)

def test_agent():
    env = gym.make('CartPole-v0')
    env = PPOObserver(env)
    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape, 128)

    actor_critic = ActorCritic(env.observation_space, env.action_space)
    master = PPOMaster(actor_critic)

    data = run(env, actor_critic, replay_buffer)
    loss_pi, loss_v = master.update(data)

    env = gym.make('Pendulum-v0')
    env = PPOObserver(env)
    replay_buffer = ReplayBuffer(env.observation_space.shape,
                                 env.action_space.shape, 128)

    actor_critic = ActorCritic(env.observation_space, env.action_space)
    master = PPOMaster(actor_critic)

    data = run(env, actor_critic, replay_buffer)
    loss_pi, loss_v = master.update(data)