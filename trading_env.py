"""
TradingEnv - Gymnasium environment.
Action space: 0=Hold, 1=Buy, 2=Sell
State: normalized features + [position_flag, cash_ratio, holdings_ratio]
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List


class TradingEnv(gym.Env):
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001, 
        feature_columns: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.n_steps = len(self.df)
        
        # Auto-detect normalized columns if not specified
        if feature_columns is None:
            self.feature_columns = [c for c in df.columns if '_norm' in c]
        else:
            self.feature_columns = feature_columns
        
        # State = market features + 3 portfolio indicators
        self.state_dim = len(self.feature_columns) + 3
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.portfolio_value = initial_balance
        self.done = False
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.done = False
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        # Market features
        market_feats = self.df.iloc[self.current_step][self.feature_columns].values
        
        # Portfolio features
        current_price = self.df.iloc[self.current_step]['Close']
        position_flag = 1.0 if self.shares_held > 0 else 0.0
        cash_ratio = self.balance / self.initial_balance
        holdings_value = (self.shares_held * current_price) / self.initial_balance
        
        return np.concatenate([
            market_feats, 
            [position_flag, cash_ratio, holdings_value]
        ]).astype(np.float32)
    
    def _get_portfolio_value(self, price: Optional[float] = None) -> float:
        if price is None:
            price = self.df.iloc[self.current_step]['Close']
        return self.balance + (self.shares_held * price)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_price = self.df.iloc[self.current_step]['Close']
        prev_portfolio = self.portfolio_value
        prev_shares = self.shares_held
        
        # Execute action
        if action == 1 and self.shares_held == 0 and self.balance > 0:
            # Buy
            cost_per_share = current_price * (1 + self.transaction_cost)
            shares_to_buy = int(self.balance / cost_per_share)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * cost_per_share
                self.shares_held = shares_to_buy
                
        elif action == 2 and self.shares_held > 0:
            # Sell
            sale_value = self.shares_held * current_price * (1 - self.transaction_cost)
            self.balance += sale_value
            self.shares_held = 0
        
        # Advance time
        self.current_step += 1
        self.done = self.current_step >= self.n_steps - 1
        
        # Calculate new portfolio value
        new_portfolio = self._get_portfolio_value()
        self.portfolio_value = new_portfolio
        
        # Reward: percentage return scaled by 100
        if prev_portfolio > 0:
            reward = ((new_portfolio - prev_portfolio) / prev_portfolio) * 100
        else:
            reward = 0.0
        
        # Penalty for invalid actions
        if action == 1 and prev_shares > 0:
            reward = -0.1
        elif action == 2 and prev_shares == 0:
            reward = -0.1
        
        info = {
            'portfolio_value': new_portfolio,
            'price': current_price,
            'shares': self.shares_held,
            'cash': self.balance,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, self.done, False, info
