import numpy as np

def calculate_simple_reward(previous_balance, current_balance):
    """
    Thưởng dựa trên biến động số dư (PnL ròng).
    """
    return current_balance - previous_balance

def shape_reward(pnl, trade_closed=False, is_win=False, fee_paid=0.0):
    """
    Hàm thưởng tối ưu cho Swing Trading:
    1. Thưởng PnL thực tế.
    2. Phạt nặng phí giao dịch để AI không spam lệnh.
    3. Thưởng thêm khi chốt lời thành công.
    """
    # Gốc vẫn là PnL (Lợi nhuận thực tế)
    reward = pnl
    
    # 1. PHẠT PHÍ GIAO DỊCH: Đây là quan trọng nhất để dừng spam lệnh
    if fee_paid > 0:
        reward -= (fee_paid * 2.0) # Phạt gấp đôi phí để AI trân trọng mỗi lần vào lệnh
    
    # 2. LOGIC CHỐT LỆNH (Chỉ thưởng/phạt khi thực sự đóng vị thế)
    if trade_closed:
        if is_win:
            reward += 1.5  # Thưởng thêm 1.5 điểm nếu chốt lời
        else:
            reward -= 2.0  # Phạt nặng 2.0 điểm nếu để lỗ (Cắt lỗ chậm)
            
    # 3. PHẠT GIỮ LỆNH QUÁ LÂU (Holding Penalty)
    # Thay vì phạt pnl == 0, ta chỉ phạt một lượng cực nhỏ (0.001) mỗi bước 
    # để AI không "ngâm" lệnh quá 1 tuần mà không làm gì.
    reward -= 0.001 
    
    return reward
