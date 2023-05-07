# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym

class DiscretePortfolioEnvFaster(gym.Env):
    """
    Gym environment to simulate portfolio management.
        - State: observation is a 'window_size' length history of returns for each asset
        - Action: each discrete action represents buying or selling 'order_size*self.current_balance' worth of any asset
        - Reward: return of the portfolio
    
    Input format: returns_df must have a numerical index, a 'Date' column as the first column (dtype=datetime64), and a column for each asset.
    
    Note: For test mode, set episode_length=-1
    """
    
    def __init__(self, df, window_size=14, order_size=0.1, starting_balance=1000, episode_length=90):
        
        # column names
        self.NUM_ASSETS = len(df.columns)
        
        # environment constants
        self.WINDOW_SIZE = window_size
        self.ORDER_SIZE = order_size
        self.PORTFOLIO_PRECISION = len(str(self.ORDER_SIZE).split('.')[-1]) # number of decimal places of order_size
        self.STARTING_BALANCE = starting_balance
        self.EPISODE_LENGTH = episode_length
        
        # create dataframe
        self.df = df
        
        # initialize action/observation space
        self.action_space = gym.spaces.Discrete(self.NUM_ASSETS*2 + 1) # buy/sell for each stock or do nothing
        self.observation_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (self.WINDOW_SIZE*self.NUM_ASSETS,), 
            dtype = np.float64
        )
        
        # reset the environment (this also creates many other variables and columns)
        self.reset()
    
        
    def reset(self):
        """
        Reset the environment to a randomly chosen starting index.
        """
        
        # reset current information
        self.current_balance = self.STARTING_BALANCE
        self.current_portfolio = np.append(np.zeros(self.NUM_ASSETS), 1.0) # e.g [0, 0, 0, 1] for 3 assets
        self.current_length = 0
        self.current_index = self.df.index.get_loc(self.df[self.WINDOW_SIZE:-self.EPISODE_LENGTH].sample().index[0]) # random starting index
        
        # return current observation
        return self.get_observation()
    
    
    def get_observation(self):
        """
        Return a WINDOW_SIZE day history of returns (including the returns of the current index).
        TODO check if iloc makes it peek into the future
        """
        return self.df.iloc[self.current_index-self.WINDOW_SIZE+1:self.current_index+1].to_numpy().flatten()
    
    
    def step(self, action):

        self.current_index += 1
        self.current_length += 1
        
        if self.current_length >= self.EPISODE_LENGTH:
            done = True
        else:
            done = False

        action -= self.NUM_ASSETS # convert the action to a number between -len(ASSETS) and +len(ASSETS)
        action_asset, action_sign = abs(action), np.sign(action)
        if (action_sign>0) and (self.current_portfolio[-1]>0): # if we want to buy and have cash
            self.current_portfolio[action_asset-1] += self.ORDER_SIZE
            self.current_portfolio[-1] -= self.ORDER_SIZE
        elif (action_sign<0) and (self.current_portfolio[action_asset-1]>0): # if we want to sell and have the asset
            self.current_portfolio[action_asset-1] -= self.ORDER_SIZE
            self.current_portfolio[-1] += self.ORDER_SIZE
        else: # if we want to do nothing TODO remove
            pass
        self.current_portfolio = self.current_portfolio.round(decimals=self.PORTFOLIO_PRECISION) # round to avoid floating point error
        
        previous_balance = self.current_balance
        incoming_returns = self.df.iloc[self.current_index]
        self.current_balance *= self.current_portfolio[-1] + ((1+incoming_returns) * self.current_portfolio[:-1]).sum()
        
        reward = (self.current_balance - previous_balance) / previous_balance
        
        observation = self.get_observation()
        
        return observation, reward, done, {}
    
    def render(self):
        pass

# Ideas: timestamp indexing, splitting into subfunctions, loc vs iloc vs indexing