%%writefile /content/AI_SWING/env/sim_futures_env.py
import numpy as np
import pandas as pd
from env.reward_func import shape_reward

class SimFuturesEnv:
    def __init__(self, df, initial_balance=100, leverage=3, commission=0.0004):
        self.df = df
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.reset()

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_balance
        self.position = 0  # 0: None, 1: Long, 2: Short (Khớp với Agent)
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.df.iloc[self.current_step].values

    def step(self, action):
        # Action Map: 0: Short, 1: Hold (None), 2: Long
        if self.done: return self._get_state(), 0, True, {}

        prev_capital = self.capital
        current_price = self.df.iloc[self.current_step]['close']
        fee_paid = 0.0
        trade_closed = False

        # 1. LOGIC GIAO DỊCH
        # Nếu đang có lệnh mà action khác đi -> Đóng lệnh cũ
        if self.position != 0 and action != self.position:
            # Tính toán PnL khi đóng lệnh
            if self.position == 2: # Đóng Long
                pnl_percent = (current_price - self.entry_price) / self.entry_price
            elif self.position == 0: # Đóng Short
                pnl_percent = (self.entry_price - current_price) / self.entry_price
            
            pnl_amount = pnl_percent * self.capital * self.leverage
            self.capital += pnl_amount
            
            # Trừ phí đóng lệnh
            fee = self.capital * self.commission
            self.capital -= fee
            fee_paid += fee
            
            self.position = 0
            trade_closed = True

        # Nếu đang trống lệnh mà muốn vào lệnh mới
        if self.position == 0 and action != 1:
            self.position = action
            self.entry_price = current_price
            # Trừ phí mở lệnh
            fee = self.capital * self.commission
            self.capital -= fee
            fee_paid += fee

        # 2. DI CHUYỂN BƯỚC THỜI GIAN
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # 3. TÍNH TOÁN PHẦN THƯỞNG (REWARD)
        current_pnl = self.capital - prev_capital
        reward = shape_reward(
            pnl=current_pnl, 
            trade_closed=trade_closed, 
            is_win=(current_pnl > 0),
            fee_paid=fee_paid
        )

        # Bảo vệ tài khoản: Nếu cháy túi thì dừng
        if self.capital <= 0:
            self.capital = 0
            self.done = True

        return self._get_state(), reward, self.done, {"capital": self.capital}
