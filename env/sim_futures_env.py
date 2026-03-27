# env/sim_futures_env.py
import numpy as np

class FuturesTradingEnv:
    """
    A simplified simulation environment for futures trading.
    This class mimics the basic structure of an OpenAI Gym environment.
    """
    def __init__(self, df, initial_balance=10000, leverage=3, commission=0.0004):
        """
        Initialize the trading environment.
        
        Args:
            df (pd.DataFrame): DataFrame containing the market data (prices, indicators).
            initial_balance (float): The starting account balance.
            leverage (int): The leverage to use for trades.
            commission (float): The trading commission fee.
        """
        self.df = df
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        
        # State space would be the financial data at a given time step
        self.state_shape = (len(df.columns),) 
        # Action space: 0: Hold, 1: Long, 2: Short
        self.action_space_n = 3 

        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        
        Returns:
            np.array: The initial state of the environment.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: None, 1: Long, -1: Short
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Gets the state observation for the current time step.
        """
        return self.df.iloc[self.current_step].values

    def step(self, action):
        """
        Executes one time step within the environment.
        
        Args:
            action (int): The action to take (0: Hold, 1: Long, 2: Short).
            
        Returns:
            tuple: A tuple containing (next_state, reward, done, info).
        """
        if self.done:
            raise Exception("Environment has finished. Please reset.")
            
        # Placeholder for reward calculation and state transition
        # This logic would be complex, involving PnL calculation, fees, etc.
        reward = 0
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is finished
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        next_state = self._get_state()
        info = {}  # For debugging info
        
        return next_state, reward, self.done, info
