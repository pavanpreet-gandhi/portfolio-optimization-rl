from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class PortOptEnv(Env):
    def __init__(self, df, window_size=14, order_size=0.1, starting_balance=1000, episode_length=90): # done
        
        self.df = df
        self.NUM_ASSETS = len(self.df.columns)
        self.NUM_DAYS = len(self.df)
        self.WINDOW_SIZE = window_size
        self.ORDER_SIZE = order_size
        self.PORTFOLIO_PRECISION = len(str(self.ORDER_SIZE).split('.')[-1])
        self.STARTING_BALANCE = starting_balance
        self.EPISODE_LENGTH = episode_length
        
        # each action represents buying or selling ORDER_SIZE percent of a single stock
        self.action_space = Discrete(self.NUM_ASSETS*2 + 1)
        # each observation is a WINDOW_SIZE day history of returns ending with the current return
        self.observation_space = Box(
            low = 0.0, # can this be higher? TODO
            high = 2.0, # can this be lower? TODO
            shape = (self.WINDOW_SIZE*self.NUM_ASSETS,), 
            dtype = np.float64
        )
        self.reset()
        
    def reset(self):
        self.balance = self.STARTING_BALANCE # total value of the portfolio
        # percentage allocations of all the assets in the portfolio e.g [1, 0, 0, 0] for 3 assets
        self.portfolio = np.insert(np.zeros(self.NUM_ASSETS), 0, 1.0)
        self.current_length = 0
        self.current_index = self.df.index.get_loc(self.df[self.WINDOW_SIZE:-self.EPISODE_LENGTH].sample().index[0]) # random starting index
        return self.get_observation()
    
    def get_observation(self):
        return self.df.iloc[self.current_index-self.WINDOW_SIZE+1: self.current_index+1].to_numpy().flatten()
        
    def step(self, action):
        
        # register action
        action = action - self.NUM_ASSETS # e.g possible actions for 3 stocks are {-3, -2, -1, 0, 1, 2, 3}
        action_index, action_sign = abs(action), np.sign(action)
        
        # update portfolio
        if action_sign > 0 and self.portfolio[0] > 0: # we can buy as long as we have cash
            self.portfolio[action_index] += self.ORDER_SIZE
            self.portfolio[0] -= self.ORDER_SIZE
        elif action_sign < 0 and self.portfolio[action_index] > 0: # we can sell as long as we have the asset we want to sell
            self.portfolio[action_index] -= self.ORDER_SIZE
            self.portfolio[0] += self.ORDER_SIZE
        else:
            pass
        self.portfolio = self.portfolio.round(decimals=self.PORTFOLIO_PRECISION) # round to avoid floating point error
        
        # update current_index
        self.current_index += 1
        self.current_length += 1
        if self.current_length >= self.EPISODE_LENGTH:
            done = True
        else:
            done = False
        
        # update balance
        old_balance = self.balance
        incoming_returns = self.df.iloc[self.current_index]
        self.balance = self.balance * (self.portfolio[0] + (incoming_returns * self.portfolio[1:]).sum())
        if self.balance <= 0:
            done = True
        
        # compute reward
        reward = self.balance/old_balance
        
        # get new observation
        observation = self.get_observation()
        
        # return step data
        return observation, reward, done, {}
    
    def render(self):
        pass