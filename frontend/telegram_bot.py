import os
import yaml
import sqlite3
import telebot
import torch

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config/settings.yaml')
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
BOT_TOKEN = config['telegram'].get('bot_token', '')
AUTHORIZED_CHAT_ID = str(config['telegram'].get('chat_id', ''))

bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None
_ai_scan_callback = None 

def check_auth(message):
    if not AUTHORIZED_CHAT_ID or AUTHORIZED_CHAT_ID == 'None':
        return True
    return str(message.chat.id) == AUTHORIZED_CHAT_ID

if bot:
    @bot.message_handler(commands=['start', 'help'])
    def cmd_help(message):
        if not check_auth(message): return
        help_text = """
🤖 *HỆ THỐNG AI SWING ĐÃ SẴN SÀNG*

Các lệnh điều khiển:
👉 /check - Kiểm tra lệnh AI đang đề xuất/nắm giữ gần nhất.
👉 /scan - Ép AI thức dậy quét thị trường và phân tích ngay lập tức.
        """
        bot.reply_to(message, help_text, parse_mode="Markdown")

    @bot.message_handler(commands=['check'])
    def cmd_check(message):
        if not check_auth(message): return
        db_path = os.path.join(os.path.dirname(__file__), '../data/memory/memory.sqlite')
        if not os.path.exists(db_path):
            bot.reply_to(message, "⚠️ Hệ thống chưa ghi nhận tín hiệu nào.")
            return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, signal, entry, margin, leverage, sl, tp, time FROM signals ORDER BY id DESC LIMIT 1")
            last_signal = cursor.fetchone()
            conn.close()
            if last_signal:
                symbol, signal, entry, margin, leverage, sl, tp, time = last_signal
                msg = f"📌 *LỆNH GẦN NHẤT:*\n{signal} {symbol} | Entry: {entry}\nSL: {sl} | TP: {tp}"
                bot.reply_to(message, msg, parse_mode="Markdown")
            else:
                bot.reply_to(message, "Chưa có tín hiệu nào được lưu.")
        except Exception as e:
            bot.reply_to(message, f"❌ Lỗi: {e}")

    @bot.message_handler(commands=['scan'])
    def cmd_scan(message):
        if not check_auth(message): return
        bot.reply_to(message, "🔄 Đang quét thị trường...")
        try:
            if _ai_scan_callback:
                with torch.no_grad():
                    result_msg = _ai_scan_callback()
                bot.reply_to(message, result_msg, parse_mode="Markdown")
            else:
                bot.reply_to(message, "⚠️ AI callback chưa được kết nối.")
        except Exception as e:
            bot.reply_to(message, f"❌ Lỗi: {e}")

def send_telegram_alert(message_text):
    if not bot:
        return
    try:
        bot.send_message(AUTHORIZED_CHAT_ID, message_text, parse_mode="Markdown")
    except Exception as e:
        print(f"Error sending Telegram: {e}")

def start_listening(scan_func=None):
    global _ai_scan_callback
    _ai_scan_callback = scan_func
    if not bot:
        print("[BOT] Telegram disabled (no token)")
        return
    print("[BOT] Telegram Bot started.")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)

class TelegramBot:
    def __init__(self):
        pass
    
    def run_in_thread(self):
        start_listening()
    
    def send_message(self, text):
        send_telegram_alert(text)
    
    def send_signal(self, signal):
        signal_text = f"""
📌 *TÍN HIỆU MỚI:*
{signal.get('signal', 'N/A')} | Entry: {signal.get('entry_price', 'N/A')}
SL: {signal.get('sl_price', 'N/A')} | TP: {signal.get('tp_price', 'N/A')}
        """
        send_telegram_alert(signal_text)