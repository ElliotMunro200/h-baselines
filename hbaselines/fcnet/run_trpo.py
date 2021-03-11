import gym

from hbaselines.fcnet.trpo import TRPO
from hbaselines.fcnet.trpo import FeedForwardPolicy

env = gym.make('HalfCheetah-v2')

model = TRPO(FeedForwardPolicy,
             env,
             verbose=1,
             lam=0.95,
             timesteps_per_batch=2000)
model.learn(total_timesteps=1000000)
