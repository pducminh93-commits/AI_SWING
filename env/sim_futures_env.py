
import numpy as np
import pandas as pd
from env.reward_func import shape_reward

class SimFuturesEnv:
    def __init__(self, df, initial_balance=100.0, leverage=3, commission=0.0004):
        self.df = df
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.reset()

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_balance
        self.position = 1  # 0: Short, 1: Hold, 2: Long
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Trả về toàn bộ hàng dữ liệu tại bước hiện tại
        return self.df.iloc[self.current_step].values

    def step(self, action):
        if self.done: return self._get_state(), 0, True, {}

        prev_capital = self.capital
        current_price = self.df.iloc[self.current_step]['close']
        fee_paid = 0.0
        trade_closed = False

        # Logic đóng lệnh và tính PnL
        if self.position != 1 and action != self.position:
            if self.position == 2: # Long
                pnl_percent = (current_price - self.entry_price) / self.entry_price
            else: # Short
                pnl_percent = (self.entry_price - current_price) / self.entry_price
            
            self.capital += pnl_percent * self.capital * self.leverage
            fee = self.capital * self.commission
            self.capital -= fee
            fee_paid += fee
            self.position = 1
            trade_closed = True

        # Logic mở lệnh mới
        if self.position == 1 and action != 1:
            self.position = action
            self.entry_price = current_price
            fee = self.capital * self.commission
            self.capital -= fee
            fee_paid += fee

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # Tính Reward
        current_pnl = self.capital - prev_capital
        reward = shape_reward(pnl=current_pnl, trade_closed=trade_closed, 
                              is_win=(current_pnl > 0), fee_paid=fee_paid)

        if self.capital <= 0:
            self.capital = 0
            self.done = True

        return self._get_state(), reward, self.done, {"capital": self.capital}
