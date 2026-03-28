import yaml
import math
from typing import Dict, Any

class SignalParser:
    """
    Bộ phiên dịch tín hiệu AI thành thông số đặt lệnh thực tế.
    Tuân thủ tuyệt đối quy tắc: Ký quỹ cố định (Fixed Margin) và Đòn bẩy động (Dynamic Leverage).
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        trading_config = self.config['trading']
        self.capital = float(trading_config['initial_capital'])
        self.margin_pct = float(trading_config['margin_per_trade'])
        self.min_lev = int(trading_config['min_leverage'])
        self.max_lev = int(trading_config['max_leverage'])
        
        # Hệ số nới Stoploss (Ví dụ: 1.5 lần ATR để né râu nến)
        self.atr_multiplier = 1.5 
        # Tỷ lệ R:R mặc định (Ví dụ: 1:2 -> Lãi gấp đôi lỗ)
        self.rr_ratio = 2.0

    def parse_ai_action(self, action: int, current_price: float, atr_value: float) -> Dict[str, Any]:
        """
        Dịch hành động (0, 1, 2) của PPO Agent thành lệnh hoàn chỉnh.
        Quy ước Agent: 0 = SHORT, 1 = HOLD, 2 = LONG
        """
        # 1. Lọc tín hiệu Đứng ngoài
        if action == 1:
            return {"signal": "HOLD"}

        signal = "LONG" if action == 2 else "SHORT"

        # 2. Tính toán Margin cố định (Tuyệt đối không vượt ngưỡng)
        margin_usdt = self.capital * self.margin_pct

        # 3. Tính toán biên độ rủi ro (Nới rộng theo ATR hiện tại)
        sl_distance = atr_value * self.atr_multiplier
        sl_pct = sl_distance / current_price

        # 4. Tính Đòn bẩy Toán học (Ép vào mốc 10x - 20x)
        # Logic: Chọn đòn bẩy sao cho khi giá chạm Stoploss, lệnh vừa đúng cháy phần Margin 5 USDT.
        # Công thức: Leverage lý tưởng = 1 / %_Biến_động_đến_SL
        if sl_pct > 0:
            ideal_leverage = 1 / sl_pct
        else:
            ideal_leverage = self.min_lev

        # Cắt gọt (Clamp) đòn bẩy nằm gọn trong giới hạn an toàn của file settings.yaml
        leverage = max(self.min_lev, min(self.max_lev, math.floor(ideal_leverage)))

        # 5. Xác định tọa độ Cắt lỗ (SL) và Chốt lời (TP)
        tp_distance = sl_distance * self.rr_ratio

        if signal == "LONG":
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else: # SHORT
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        return {
            "signal": signal,
            "entry_price": round(current_price, 4),
            "margin_usdt": round(margin_usdt, 2),
            "leverage": int(leverage),
            "sl_price": round(sl_price, 4),
            "tp_price": round(tp_price, 4),
            "risk_reward": f"1:{self.rr_ratio}",
            "info": f"AI chọn {signal}. Biên độ ATR: {round(atr_value, 2)}"
        }