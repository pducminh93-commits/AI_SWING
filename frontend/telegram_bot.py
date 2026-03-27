import os
import yaml
import sqlite3
import telebot
import torch

# ==========================================
# 1. KHỞI TẠO VÀ BẢO MẬT
# ==========================================
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/settings.yaml')
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
BOT_TOKEN = config['telegram']['bot_token']
AUTHORIZED_CHAT_ID = str(config['telegram']['chat_id'])

# Khởi tạo Bot
bot = telebot.TeleBot(BOT_TOKEN)

# Biến toàn cục để chứa hàm suy luận AI (Callback)
# Giúp bot gọi ngược lại AI mà không bị lỗi vòng lặp import (Circular Import)
_ai_scan_callback = None 

class TelegramBot:
    def __init__(self):
        pass
    
    def run_in_thread(self):
        start_listening()
    
    def send_message(self, text):
        send_telegram_alert(text)
    
    def send_signal(self, signal):
        signal_text = f"""
📌 *TÍN HIỆU MỚI TỪ AI:*
---
{signal.get('signal', 'N/A')} {signal.get('symbol', 'N/A')}
💵 Margin: {signal.get('margin', 'N/A')} USDT
🎯 Entry: {signal.get('entry', 'N/A')}
🛑 SL: {signal.get('sl', 'N/A')}
✅ TP: {signal.get('tp', 'N/A')}
---
⏰ Time: {signal.get('time', 'N/A')}
"""
        send_telegram_alert(signal_text)

def check_auth(message):
    """Lá chắn bảo mật: Chỉ nhận lệnh từ chính chủ (Dựa vào Chat ID)"""
    return str(message.chat.id) == AUTHORIZED_CHAT_ID

# ==========================================
# 2. CÁC LỆNH ĐIỀU KHIỂN TỪ XA (COMMANDS)
# ==========================================

@bot.message_handler(commands=['start', 'help'])
def cmd_help(message):
    if not check_auth(message): return
    help_text = """
🤖 *HỆ THỐNG MOE FUTURES SWING ĐÃ SẴN SÀNG*

Các lệnh điều khiển:
👉 /check - Kiểm tra lệnh AI đang đề xuất/nắm giữ gần nhất.
👉 /scan - Ép AI thức dậy quét thị trường và phân tích ngay lập tức.
    """
    bot.reply_to(message, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['check'])
def cmd_check(message):
    """Truy xuất Database để xem AI đang báo kèo gì"""
    if not check_auth(message): return
    
    db_path = os.path.join(os.path.dirname(__file__), '../data/memory/memory.sqlite')
    
    if not os.path.exists(db_path):
        bot.reply_to(message, "⚠️ Hệ thống chưa ghi nhận tín hiệu nào (Database trống).")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Lấy kèo gần nhất từ DB (Giả định bảng tên là 'signals')
        cursor.execute("SELECT symbol, signal, entry, margin, leverage, sl, tp, time FROM signals ORDER BY id DESC LIMIT 1")
        last_signal = cursor.fetchone()
        conn.close()

        if last_signal:
            symbol, signal, entry, margin, leverage, sl, tp, time = last_signal
            msg = f"""
📌 *LỆNH GẦN NHẤT TRONG BỘ NHỚ:*
⏱ Thời gian AI chốt: {time}
---
{signal} {symbol} | Đòn bẩy: {leverage}x
💵 Margin (Ký quỹ): {margin} USDT
🎯 Entry: {entry}
🛑 Stoploss: {sl}
✅ Take Profit: {tp}
---
👉 *Hãy kiểm tra giá hiện tại trên app Binance để quyết định Cắt/Giữ.*
"""
            bot.reply_to(message, msg, parse_mode="Markdown")
        else:
            bot.reply_to(message, "Chưa có tín hiệu nào được lưu.")
            
    except sqlite3.OperationalError:
        bot.reply_to(message, "⚠️ Bảng dữ liệu chưa được khởi tạo. Hãy đợi AI quét lần đầu tiên.")
    except Exception as e:
        bot.reply_to(message, f"❌ Lỗi đọc dữ liệu: {e}")

@bot.message_handler(commands=['scan'])
def cmd_scan(message):
    """Ép AI quét dữ liệu ngay lập tức ngoài lịch trình"""
    if not check_auth(message): return
    
    bot.reply_to(message, "🔄 Đang ép AI thức dậy kéo dữ liệu và phân tích. Quá trình này mất khoảng 5-10 giây, vui lòng đợi...")
    
    try:
        if _ai_scan_callback:
            # Bật khiên bảo vệ VRAM khi quét thủ công
            with torch.no_grad():
                result_msg = _ai_scan_callback()
            
            # Dọn rác GPU ngay sau khi quét xong
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            bot.reply_to(message, result_msg, parse_mode="Markdown")
        else:
            bot.reply_to(message, "⚠️ Lỗi: Hàm suy luận AI chưa được đấu nối vào Bot.")
    except Exception as e:
        bot.reply_to(message, f"❌ Lỗi khi AI quét: {e}")

# ==========================================
# 3. HÀM CHỦ ĐỘNG GỬI TIN NHẮN (Dành cho luồng chạy tự động)
# ==========================================
def send_telegram_alert(message_text):
    """Hàm này được gọi bởi luồng main để tự động bắn tin nhắn khi có kèo"""
    try:
        bot.send_message(AUTHORIZED_CHAT_ID, message_text, parse_mode="Markdown")
    except Exception as e:
        print(f"❌ Lỗi gửi Telegram Alert: {e}")

# ==========================================
# 4. KHỞI ĐỘNG VÒNG LẶP LẮNG NGHE
# ==========================================
def start_listening(scan_func=None):
    """
    Hàm này sẽ chạy ở một luồng (Thread) riêng biệt.
    scan_func: Là hàm chứa logic kéo API Binance + Đưa vào MoE để dự đoán.
    """
    global _ai_scan_callback
    _ai_scan_callback = scan_func
    
    print("[BOT] Telegram Bot started. Listening for commands...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)