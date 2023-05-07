import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym

class PortfolioManagementEnv(gym.Env):
    """
    Gym environment with discrete action space to simulate portfolio management.
    """
    def __init__(
        self,
        df, 
        return_cols, 
        feature_cols=[], 
        window_size = 20, 
        order_size = 0.1, 
        starting_balance = 1, 
        episode_length = 180,
        drawdown_penalty_weight = 1,
        allocations_in_obs = False
    ):
        """
        Parameters:
            - `df`: Pandas dataframe with datetime index
            - `return_cols`: List of column names containing asset returns (with the first entry being the risk free returns)
            - `feature_cols`: List of column names to be used as features
            - `episode_length`: Length of each episode (-1 makes it go from start to end)
            - `window_size`: Size of lookback window
            - `order_size`: Size of step in allocations
            - `starting_balance`: Amount of cash to start with
            - `episode_length`: Length of each episode
            - `drawdown_penalty_size`: Weight of drawdown on reward
            - `allocations_in_obs`: Whether or not to include current allocations in the observation
        """
        
        # Data related constants
        self.RETURN_COLS = return_cols
        self.FEATURE_COLS = feature_cols
        self.NUM_ASSETS = len(return_cols)-1
        self.NUM_FEATURES = len(feature_cols)
        self.RETURNS = df[self.RETURN_COLS].to_numpy()
        self.FEATURES = df[self.FEATURE_COLS].to_numpy()
        self.INDEX = df.index
        
        # Environment constants
        self.WINDOW_SIZE = window_size
        self.ORDER_SIZE = order_size
        self.ALLOCATIONS_PRECISION = len(str(self.ORDER_SIZE).split('.')[-1]) # number of decimal places of order_size
        self.STARTING_BALANCE = starting_balance
        self.EPISODE_LENGTH = episode_length
        self.DRAWDOWN_PENALTY_WEIGHT = drawdown_penalty_weight
        self.ALLOCATION_IN_OBS = allocations_in_obs
        
        # Initialize action/observation space
        self.action_space = gym.spaces.Discrete(self.NUM_ASSETS*2 + 1) # buy/sell for each stock or do nothing
        if self.ALLOCATION_IN_OBS:
            self.observation_space = gym.spaces.Box(
                low = np.concatenate([self.FEATURES.min(axis=0) for _ in range(self.WINDOW_SIZE)] + [np.zeros(self.NUM_ASSETS+1)]),
                high = np.concatenate([self.FEATURES.max(axis=0) for _ in range(self.WINDOW_SIZE)] + [np.ones(self.NUM_ASSETS+1)]),
                shape = (self.WINDOW_SIZE*self.NUM_FEATURES + self.NUM_ASSETS+1,), 
                dtype = np.float64
            )
        else:
            self.observation_space = gym.spaces.Box(
                low = np.concatenate([self.FEATURES.min(axis=0) for _ in range(self.WINDOW_SIZE)]),
                high = np.concatenate([self.FEATURES.max(axis=0) for _ in range(self.WINDOW_SIZE)]),
                shape = (self.WINDOW_SIZE*self.NUM_FEATURES,), 
                dtype = np.float64
            )
        
        # Reset the environment
        self.reset()
    
        
    def reset(self):
        """
        Resets the environment to a randomly chosen starting index.
        """
        if self.EPISODE_LENGTH == -1:
            self.start_index = self.WINDOW_SIZE
        else:
            self.start_index = np.random.randint(self.WINDOW_SIZE, len(self.RETURNS)-self.EPISODE_LENGTH) # Random start index
        self.current_index = self.start_index
        
        # The allocations always adds up to 1 with starting allocations as [1, 0, 0, ..., 0] (index 0 is for cash).
        self.current_allocations = np.insert(np.zeros(self.NUM_ASSETS), 0, 1.0)
        self.current_value = self.STARTING_BALANCE
        self.weighted_cumilative_return = 0
        
        self.return_history = [0]
        self.value_history = [self.current_value]
        self.allocations_history = [self.current_allocations.copy()]
        
        return self.get_observation()
    
    
    def get_observation(self):
        """
        Returns a `WINDOW_SIZE` day history of returns and other features.
        Excludes the returns and features at the current index.
        """
        obs = self.FEATURES[self.current_index-self.WINDOW_SIZE : self.current_index].flatten()
        if self.ALLOCATION_IN_OBS:
            obs = np.concatenate((obs, self.current_allocations))
        return obs
    
    
    def update_current_allocations(self, action):
        """
        Updates the current_allocations according to the given action.
        The action can be to do nothing or to buy or sell any asset.
        An action can change up to one allocation by `order_size`.
        If an action is invalid then it is equivalent to doing nothing.
        """
        action -= self.NUM_ASSETS # Convert the action to a number between -len(ASSETS) and +len(ASSETS)
        action_asset, action_sign = abs(action), np.sign(action)
        
        # If we want to do nothing
        if action_sign==0:
            return # exit the function
        
        # If we want to buy and have cash (e.g action +3 means we want to buy the asset at position 3).
        elif (action_sign>0) and (self.current_allocations[0]>0):
            self.current_allocations[action_asset] += self.ORDER_SIZE
            self.current_allocations[0] -= self.ORDER_SIZE
        
        # If we want to sell and have the asset (e.g -1 means we want to sell asset at position 1).
        elif (action_sign<0) and (self.current_allocations[action_asset]>0):
            self.current_allocations[action_asset] -= self.ORDER_SIZE
            self.current_allocations[0] += self.ORDER_SIZE
        
        # Round to avoid floating point error
        self.current_allocations = self.current_allocations.round(decimals=self.ALLOCATIONS_PRECISION)
    
    
    def update_current_value(self):
        """
        Updates the `current_value` according to the `current_allocations` and the incoming returns at the current index.
        Returns the previous value for return calculations.
        """
        previous_value = self.current_value
        self.current_value *= ((1+self.RETURNS[self.current_index])*self.current_allocations).sum()
        return previous_value
    
    
    def step(self, action):
        """
        Takes a step in the environment by performing the following steps:
            1. Increment the current_index
            2. Update `current_allocations` according to the given action
            3. Update `current_value` according to `current_allocations` and the incoming returns
            5. Compute the return
            6. Return (observation, reward=return, done, info)
        """
        self.current_index += 1
        
        if self.EPISODE_LENGTH == -1:
            done = bool(self.current_index >= len(self.RETURNS)-1)
        else:
            done = bool(self.current_index - self.start_index >= self.EPISODE_LENGTH)
        
        self.update_current_allocations(action)
        previous_value = self.update_current_value()
        ret = (self.current_value - previous_value) / previous_value
        if ret > 0:
            self.weighted_cumilative_return = (1 + self.weighted_cumilative_return) * (1 + ret) - 1
        else:
            self.weighted_cumilative_return = (1 + self.weighted_cumilative_return) * (1 + self.DRAWDOWN_PENALTY_WEIGHT * ret) - 1
        
        reward = self.weighted_cumilative_return * (self.current_index - self.start_index)/self.EPISODE_LENGTH
        observation = self.get_observation()

        self.return_history.append(ret)
        self.value_history.append(self.current_value)
        self.allocations_history.append(self.current_allocations)
        
        return observation, reward, done, {} # {} is a dummy variable for info
    
    def render(self, ax=None, title=''):
        """
        Renders the changing portfolio value over time as a stackplot.
        """
        value_history_array = np.array(self.value_history).reshape(-1, 1)
        allocations_history_array = np.array(self.allocations_history)
        value_breakdown = (value_history_array * allocations_history_array).transpose()
        
        if ax==None:
            plt.figure(figsize=(12,4))
            ax = plt.axes()
        
        ax.set_title(title)
        ax.stackplot(
            self.INDEX[self.start_index : self.current_index+1], 
            value_breakdown, 
            labels = [col_name.split('_')[0] for col_name in self.RETURN_COLS],
        );
        plt.gcf().autofmt_xdate();

    
    def get_portfolio_returns(self):
        """
        Returns a datetime indexed series of portfolio returns.
        """
        return pd.Series(
            self.return_history, 
            index=self.INDEX[self.start_index : self.current_index+1]
        )