# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym

class DiscretePortfolioEnv(gym.Env):
    """
    Gym environment to simulate portfolio management.
        - State: observation is a 'window_size' length history of returns for each asset
        - Action: each discrete action represents buying or selling 'order_size*self.current_balance' worth pf any asset
        - Reward: return of the portfolio
    
    Input format: returns_df must have a numerical index, a 'Date' column as the first column (dtype=datetime64), and a column for each asset.
    """
    
    def __init__(self, returns_df, window_size=14, order_size=0.1, starting_balance=1000, episode_length=90):
        
        # column names
        self.DATE = 'Date'
        self.BALANCE = 'BALANCE'
        self.ASSETS = list(returns_df.columns)[1:]
        self.ALLOCATION_COLUMNS = [f'{asset}_ALLOCATION' for asset in self.ASSETS]
        
        # environment constants
        self.WINDOW_SIZE = window_size
        self.ORDER_SIZE = order_size
        self.PORTFOLIO_PRECISION = len(str(self.ORDER_SIZE).split('.')[-1]) # number of decimal places of order_size
        self.STARTING_BALANCE = starting_balance
        self.EPISODE_LENGTH = episode_length
        
        # create dataframe
        self.df = returns_df.copy()
        
        # initialize action/observation space
        self.action_space = gym.spaces.Discrete(len(self.ASSETS)*2 + 1) # buy/sell for each stock or do nothing
        self.observation_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (self.WINDOW_SIZE*len(self.ASSETS),), 
            dtype = np.float64
        )
        
        # reset the environment (this also creates many other variables and columns)
        self.reset()
    
        
    def reset(self):
        """
        Reset the environment to a randomly chosen starting index.
        """
        # reset allocations and balance columns
        self.df[self.ALLOCATION_COLUMNS] = np.NaN
        self.df[self.BALANCE] = np.NaN
        
        # reset current information
        self.current_balance = self.STARTING_BALANCE
        self.current_portfolio = np.append(np.zeros(len(self.ASSETS)), 1.0) # e.g [0, 0, 0, 1] for 3 assets
        self.current_index = self.df[self.WINDOW_SIZE:-self.EPISODE_LENGTH].sample().index[0] # random starting index
        self.current_length = 0
        
        # update dataframe
        self.update_dataframe()
        
        # return current observation
        return self.get_observation()
    
    
    def update_dataframe(self):
        """
        Update the BALANCE and ASSET_ALLOCATION columns with the current information.
        """
        self.df.loc[self.current_index, self.BALANCE] = self.current_balance
        self.df.loc[self.current_index, self.ALLOCATION_COLUMNS] = self.current_portfolio[:-1]
    
    
    def get_observation(self):
        """
        Return a WINDOW_SIZE day history of returns (including the returns of the current index).
        """
        return self.df[self.ASSETS][self.current_index-self.WINDOW_SIZE+1:self.current_index+1].to_numpy().flatten()
    
    
    def update_current_portfolio(self, action):
        """
        Update the current_portfolio according to the given action.
        The action can be to buy or sell 'ORDER_SIZE*current_balance' of any asset (or to do nothing).
        """
        action -= len(self.ASSETS) # convert the action to a number between -len(ASSETS) and +len(ASSETS)
        action_asset, action_sign = abs(action), np.sign(action)

        # if we want to buy and have cash
        if (action_sign>0) and (self.current_portfolio[-1]>0):
            self.current_portfolio[action_asset-1] += self.ORDER_SIZE
            self.current_portfolio[-1] -= self.ORDER_SIZE
        
        # if we want to sell and have the asset
        if (action_sign<0) and (self.current_portfolio[action_asset-1]>0):
            self.current_portfolio[action_asset-1] -= self.ORDER_SIZE
            self.current_portfolio[-1] += self.ORDER_SIZE
        
        # round to avoid floating point error
        self.current_portfolio = self.current_portfolio.round(decimals=self.PORTFOLIO_PRECISION)
    
    
    def update_current_balance(self):
        """
        Update the current_balance according to the current_portfolio and the returns at the current index.
        Return the previous balance.
        """
        previous_balance = self.current_balance
        self.current_balance *= self.current_portfolio[-1] + ((self.df.loc[self.current_index, self.ASSETS]+1)*self.current_portfolio[:-1]).sum()
        return previous_balance
    
    
    def compute_reward(self, previous_balance):
        """
        Define and return the rweard as the daily return of the portfolio.
        """
        return (self.current_balance-previous_balance)/previous_balance
    
    
    def step(self, action):
        """
        Take a step in the environment by performing the following steps:
            1. Increment index
            2. Update current_portfolio according to action
            3. Update current_balance according to current_portfolio
            4. Update dataframe according to current information
            5. Compute reward
            6. Return (observation, reward, done, info)
        """
        self.current_index += 1
        self.current_length += 1
        done = bool(self.current_length >= self.EPISODE_LENGTH)
        self.update_current_portfolio(action)
        previous_balance = self.update_current_balance()
        self.update_dataframe()
        reward = self.compute_reward(previous_balance)
        observation = self.get_observation()
        return observation, reward, done, {} # {} is a dummy dictionary for info
    
    
    def render(self):
        """
        Render the changing balance over time as a stackplot.
        """
        balance_breakdown = []
        for allocation in self.ALLOCATION_COLUMNS:
            balance_breakdown.append(self.df[allocation] * self.df.BALANCE)
        balance_breakdown.append(self.df[self.BALANCE] * (1-self.df[self.ALLOCATION_COLUMNS].sum(axis=1))) # value of balance as cash
        
        plt.figure(figsize=(10, 5));
        plt.grid();
        plt.title('Portfolio Value');
        plt.ylabel('Value ($)')
        plt.xlabel('Date')
        plt.stackplot(self.df[self.DATE], balance_breakdown, alpha=0.8, labels=self.ASSETS + ['CASH']);
        plt.legend(loc='upper left');
        plt.gcf().autofmt_xdate();