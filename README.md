# AI SWING - Trading Signal Generator

Hệ thống tạo tín hiệu giao dịch swing trading trên Binance Futures sử dụng AI kết hợp nhiều mô hình Expert.

## Cấu Trúc Thư Mục

```
AI_SWING-main/
├── ai_core/                       # Lõi AI - Mô hình và thuật toán học máy
│   ├── experts/                   # Các mô hình Expert
│   │   ├── risk_vol_mlp.py       # MLP: Risk & Volatility analysis
│   │   ├── momentum_cnn.py        # CNN: Momentum pattern recognition
│   │   └── macro_trend_gru.py     # GRU: Macro trend prediction
│   ├── rl_agent.py                # Reinforcement Learning Agent
│   ├── gating_network.py          # Gating Network - combine experts
│   ├── signal_parser.py           # Parse & aggregate signals
│   └── memory_buffer.py           # Experience replay buffer
│
├── pipeline/                      # Pipeline xử lý dữ liệu
│   ├── fetchers/
│   │   └── binance_futures.py     # Fetch dữ liệu từ Binance
│   ├── indicators/
│   │   ├── volatility.py          # ATR, Bollinger Bands
│   │   ├── trend.py               # MA, EMA, Ichimoku
│   │   ├── momentum.py            # RSI, MACD, Stochastic
│   │   └── order_flow.py          # Order flow analysis
│   └── processors/
│       ├── tensor_builder.py      # Build features tensor
│       └── timeframe_sync.py      # Multi-timeframe sync
│
├── env/                           # RL Environment
│   ├── sim_futures_env.py         # Simulated trading environment
│   └── reward_func.py             # Custom reward function
│
├── frontend/                      # Telegram Bot
│   └── telegram_bot.py            # Nhận & gửi tín hiệu trade
│
├── utils/                         # Utilities
│   └── logger.py                  # Logging configuration
│
└── docs/                          # Tài liệu
    └── performance_and_scaling.md # Hiệu năng & scaling
```

## Cài Đặt

```bash
# Clone repository
git clone https://github.com/pducminh93-commits/AI_SWING.git
cd AI_SWING-main

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Copy file cấu hình mẫu
cp .env_example .env
```

## Cấu Hình

Chỉnh sửa file `.env` để thêm API keys Binance:

```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
TELEGRAM_BOT_TOKEN=your_bot_token
```

## Cách Chạy

### Sinh tín hiệu giao dịch

```bash
python run_signals.py
```

Script sẽ:
1. Fetch dữ liệu BTCUSDT khung 1H
2. Tính toán tất cả indicators
3. Chạy qua các expert models
4. Gating network combine signals
5. Output: BUY/SELL/NEUTRAL + Entry, TP, SL

### Training AI Model

```bash
python build_data.py
```

## Hướng Dẫn Sử Dụng Chi Tiết

### Bước 1: Cài đặt môi trường

```bash
# Tạo và kích hoạt virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt packages
pip install -r requirements.txt
```

### Bước 2: Cấu hình API

```bash
# Tạo file .env từ template
copy .env_example .env
```

Chỉnh sửa `.env`:
- Nếu test: Không cần API key (dùng dữ liệu public)
- Nếu live: Thêm Binance API Key và Secret

### Bước 3: Chạy sinh tín hiệu

```bash
python run_signals.py
```

**Output mẫu:**
```
🔔 Signal: BUY
📊 Entry: 42500.00
🎯 TP1: 43000.00 (1.2%)
🎯 TP2: 43500.00 (2.4%)
🎯 TP3: 44000.00 (3.5%)
🛡️ SL: 42000.00 (-1.2%)
```

### Bước 4: Áp dụng vào Binance Futures

1. Mở Binance Futures → chọn BTCUSDT
2. Đặt **Limit Order** tại giá Entry
3. Đặt **Take Profit** tại TP1, TP2, TP3
4. Đặt **Stop Loss** tại SL

### Bước 5: Training Model (Optional)

```bash
# Build training data
python build_data.py

# Train với RL agent (cấu hình trong code)
```

### Sử dụng Telegram Bot

```bash
# Chạy bot nhận tín hiệu
python frontend/telegram_bot.py
```

Cấu hình bot:
- Thêm BOT_TOKEN trong .env
- Set webhook hoặc chạy polling

### Tùy chỉnh Parameters

Chỉnh sửa trong `run_signals.py`:
- `SYMBOL`: Cặp giao dịch (mặc định: BTCUSDT)
- `TIMEFRAME`: Khung thời gian (1H, 4H, 1D)
- `RISK_PERCENT`: % vốn rủi ro mỗi lệnh

## Tín Hiệu Giao Dịch

Output bao gồm:
- **Signal**: BUY / SELL / NEUTRAL
- **Entry**: Giá vào lệnh đề xuất
- **TP1, TP2, TP3**: Các mức Take Profit
- **SL**: Stop Loss

## Giải Thuật

1. **Expert Models**: 3 mô hình chuyên biệt (MLP, CNN, GRU)
2. **Gating Network**: Học trọng số combine các expert
3. **RL Agent**: Policy gradient cho quyết định trading
4. **Memory Buffer**: Experience replay cho training

## Checklist Tiến Độ Dự Án

- [x] Phân tích yêu cầu dự án
- [x] Thiết kế kiến trúc hệ thống
- [x] Xây dựng core AI (gating, memory, RL agent, parser)
- [x] Phát triển các expert models (GRU, CNN, MLP)
- [x] Xây dựng pipeline xử lý dữ liệu
- [x] Tạo môi trường mô phỏng và reward function
- [x] Tích hợp frontend (Telegram bot)
- [x] Huấn luyện và lưu mô hình
- [x] Kiểm thử và đánh giá hiệu năng
- [x] Viết tài liệu hướng dẫn

## Disclaimer

⚠️ **Cảnh báo**: Giao dịch cryptocurrency có rủi ro cao. Tín hiệu từ hệ thống chỉ mang tính tham khảo. Hãy nghiên cứu kỹ và quản lý rủi ro phù hợp.

## Yêu Cầu

- Python 3.8+
- Binance Futures account
- Python packages trong requirements.txt