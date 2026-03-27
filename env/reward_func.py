# env/reward_func.py

def calculate_simple_reward(previous_balance, current_balance, position_duration):
    """
    A simple reward function based on profit and loss (PnL).
    
    Args:
        previous_balance (float): Account balance at the previous time step.
        current_balance (float): Account balance at the current time step.
        position_duration (int): Number of time steps the current position has been held.
        
    Returns:
        float: The calculated reward.
    """
    pnl = current_balance - previous_balance
    
    # Simple reward is just the PnL
    reward = pnl
    
    return reward

def calculate_sharpe_ratio_reward(returns, risk_free_rate=0.0):
    """
    A reward function based on the Sharpe ratio of returns.
    This encourages high returns while penalizing high volatility.
    
    Args:
        returns (list or np.array): A series of returns from trades.
        risk_free_rate (float): The risk-free rate of return.
        
    Returns:
        float: The Sharpe ratio, which can be used as a reward.
    """
    if len(returns) < 2:
        return 0
        
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0 # Avoid division by zero
        
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio

def shape_reward(pnl, holding_penalty=-0.01, win_bonus=1.0, loss_penalty=-1.5):
    """
    Shapes the reward to encourage desired agent behavior.
    
    Args:
        pnl (float): The profit or loss for the current step.
        holding_penalty (float): A small penalty for each step a position is held.
        win_bonus (float): A bonus applied when a trade is closed with a profit.
        loss_penalty (float): A penalty applied when a trade is closed with a loss.
        
    Returns:
        float: The shaped reward.
    """
    reward = pnl
    
    # Apply a small penalty for just holding a position to encourage action
    if pnl == 0: # Assuming no PnL means holding
        reward += holding_penalty
        
    # Apply bonuses or penalties for closing trades (this logic would be in the env)
    # e.g., if trade_closed:
    #   if pnl > 0:
    #       reward += win_bonus
    #   else:
    #       reward += loss_penalty
           
    return reward
