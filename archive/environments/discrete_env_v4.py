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
        drawdown_penalty_factor = 0,
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
            - `drawdown_penalty_factor`: Weight of drawdown on reward
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
        self.DRAWDOWN_PENALTY_FACTOR = drawdown_penalty_factor
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
        self.max_value = self.STARTING_BALANCE
        
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
    
    
    def get_new_allocations(self, action, current_allocations):
        """
        Returns new allocations given the current allocations and an action.
        The action can be to do nothing or to buy or sell any asset.
        An action can change up to one allocation by `order_size`.
        If an action is invalid then it is equivalent to doing nothing.
        """
        action -= self.NUM_ASSETS # convert the action to a number between -len(ASSETS) and +len(ASSETS)
        action_asset, action_sign = abs(action), np.sign(action)
        
        if action_sign==0: # if we want to do nothing
            pass
        
        # If we want to buy and have cash (e.g action +3 means we want to buy the asset at position 3).
        elif (action_sign>0) and (current_allocations[0]>0):
            current_allocations[action_asset] += self.ORDER_SIZE
            current_allocations[0] -= self.ORDER_SIZE
        
        # If we want to sell and have the asset (e.g -1 means we want to sell asset at position 1).
        elif (action_sign<0) and (current_allocations[action_asset]>0):
            current_allocations[action_asset] -= self.ORDER_SIZE
            current_allocations[0] += self.ORDER_SIZE
        
        return current_allocations
    
    
    def get_new_value(self, current_allocations, current_value):
        """
        Returns the new portfolio value given the current value and current allocations.
        Uses the incoming returns at the current index to compute this value.
        """
        return current_value * ((1+self.RETURNS[self.current_index])*current_allocations).sum()
    
    
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
        
        self.current_allocations = self.get_new_allocations(action, self.current_allocations)
        self.current_value = self.get_new_value(self.current_allocations, self.current_value)
        self.max_value = self.current_value if self.current_value > self.max_value else self.max_value
        
        drawdown = (self.current_value / self.max_value) - 1
        reward = (self.current_value + self.DRAWDOWN_PENALTY_FACTOR * drawdown)
        reward *= (self.current_index - self.start_index) / self.EPISODE_LENGTH # scale with episode length
        
        observation = self.get_observation()
        
        self.value_history.append(self.current_value)
        self.allocations_history.append(self.current_allocations)
        
        return observation, reward, done, {} # {} is a dummy variable for info

    
    def get_portfolio_values(self):
        """
        Returns a datetime indexed series of portfolio value ofver time.
        """
        return pd.Series(
            self.value_history, 
            index=self.INDEX[self.start_index : self.current_index+1]
        )
    
    
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