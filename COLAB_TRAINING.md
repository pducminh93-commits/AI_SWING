# Hướng Dẫn Training AI SWING Trên Google Colab

## Bước 1: Upload Code Lên Colab

1. Tạo folder Google Drive: `AI_SWING`
2. Upload toàn bộ project vào folder đó
3. Mở Google Colab: https://colab.research.google.com/
4. Tạo Notebook mới → Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/AI_SWING
```

## Bước 2: Cài Đặt Môi Trường

```python
# Cài đặt dependencies
!pip install -r requirements.txt

# Cài đặt TA-Lib (cần thiết cho indicators)
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip install TA-Lib
```

## Bước 3: Chuẩn Bị Dữ Liệu Training

```python
# Tạo file .env (chỉ cần Binance API để lấy dữ liệu)
import os
os.environ['BINANCE_API_KEY'] = 'YOUR_KEY'
os.environ['BINANCE_SECRET_KEY'] = 'YOUR_SECRET'

# Chạy build data
!python build_data.py
```

**Output:** File `data/processed/data_futures_swing.parquet`

---

# PHẦN 1: TRAINING EXPERTS (Các Chuyên Gia)

## Expert 1: RiskVolatilityMLP (Phân Tích Rủi Ro)

```python
import torch
import torch.nn as nn
from ai_core.experts.risk_vol_mlp import RiskVolatilityMLP
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet('data/processed/data_futures_swing.parquet')

# Chuẩn bị features và labels (label: 0=low_risk, 1=medium, 2=high_risk)
X = df.drop(['symbol_name'], axis=1, errors='ignore').values
y = df['volatility_label']  # Cần tạo label từ ATR volatility

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RiskVolatilityMLP(input_dim=X.shape[1], output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(50):
    model.train()
    batch_X = torch.FloatTensor(X_train).to(device)
    batch_y = torch.LongTensor(y_train).to(device)
    
    optimizer.zero_grad()
    output = model(batch_X, training=True)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Lưu model
torch.save(model.state_dict(), 'models/risk_vol_mlp.pth')
print("✅ MLP Expert training done!")
```

## Expert 2: MomentumCNN (Nhận Dạng Hình Thái Nến)

```python
import torch
import torch.nn as nn
from ai_core.experts.momentum_cnn import MomentumCNNExpert

# Dữ liệu dạng sequence cho CNN
seq_len = 30
X_seq = []  # Tạo sequences từ dữ liệu
for i in range(seq_len, len(df)):
    X_seq.append(X[i-seq_len:i])
X_seq = torch.FloatTensor(X_seq)

# Model
model = MomentumCNNExpert(input_channels=X.shape[1], seq_len=seq_len, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(50):
    model.train()
    # Reshape cho CNN: (batch, channels, seq_len)
    x_input = X_seq[:len(X_train)].permute(0, 2, 1).to(device)
    y_input = torch.LongTensor(y_train[:len(X_train)]).to(device)
    
    optimizer.zero_grad()
    output = model(x_input)
    loss = criterion(output, y_input)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'models/momentum_cnn.pth')
print("✅ CNN Expert training done!")
```

## Expert 3: MacroTrendGRU (Dự Báo Xu Hướng Dài Hạn)

```python
from ai_core.experts.macro_trend_gru import MacroTrendGRU

# Model GRU
model = MacroTrendGRU(
    input_dim=X.shape[1],
    hidden_dim=128,
    n_layers=2,
    output_dim=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training với sequences
h = model.init_hidden(batch_size=32, device=device)

for epoch in range(50):
    model.train()
    # Input shape: (batch, seq_len, features)
    x_input = X_seq[:32].to(device)
    
    optimizer.zero_grad()
    output, h = model(x_input, h)
    # Tính loss với target (giá thay đổi)
    loss = criterion(output.squeeze(), torch.zeros(32).to(device))  # Demo
    loss.backward()
    optimizer.step()
    h = h.detach()

torch.save(model.state_dict(), 'models/macro_trend_gru.pth')
print("✅ GRU Expert training done!")
```

---

# PHẦN 2: TRAINING GATING NETWORK + RL AGENT

```python
import torch
from ai_core.gating_network import GatingNetwork
from ai_core.rl_agent import RLAgent
from ai_core.memory_buffer import MemoryBuffer
import pandas as pd

# Load data
df = pd.read_parquet('data/processed/data_futures_swing.parquet')
X = df.drop(['symbol_name'], axis=1, errors='ignore').values

# Khởi tạo experts
from ai_core.experts.risk_vol_mlp import RiskVolatilityMLP
from ai_core.experts.momentum_cnn import MomentumCNNExpert
from ai_core.experts.macro_trend_gru import MacroTrendGRU

experts = [
    RiskVolatilityMLP(input_dim=X.shape[1], output_dim=2).to(device),
    MomentumCNNExpert(input_channels=X.shape[1], num_classes=2).to(device),
    MacroTrendGRU(input_dim=X.shape[1], hidden_dim=64, n_layers=2, output_dim=1).to(device)
]

# Load pretrained weights
experts[0].load_state_dict(torch.load('models/risk_vol_mlp.pth'))
experts[1].load_state_dict(torch.load('models/momentum_cnn.pth'))
experts[2].load_state_dict(torch.load('models/macro_trend_gru.pth'))

# Freeze experts (chỉ train gating)
for exp in experts:
    for p in exp.parameters():
        p.requires_grad = False

# Gating Network
gating = GatingNetwork(num_experts=3, expert_features=X.shape[1]).to(device)
optimizer_gating = torch.optim.Adam(gating.parameters(), lr=0.0005)

# RL Agent
agent = RLAgent(state_dim=X.shape[1], action_dim=3).to(device)
optimizer_agent = torch.optim.Adam(agent.parameters(), lr=0.001)

# Memory Buffer
memory = MemoryBuffer(capacity=10000)

# Training loop
for epoch in range(100):
    for i in range(0, len(X), 32):
        batch_x = torch.FloatTensor(X[i:i+32]).to(device)
        
        # Get expert outputs
        expert_outputs = []
        for exp in experts:
            exp.eval()
            with torch.no_grad():
                out = exp(batch_x)
                expert_outputs.append(out)
        
        # Gating weights
        weights = gating(batch_x, expert_outputs)
        
        # RL Agent decision
        state = torch.cat([batch_x, weights], dim=1)
        action_probs = agent(state)
        
        # Store in memory & update
        reward = compute_reward(batch_x)  # Hàm reward tùy chỉnh
        memory.push(state, action_probs.argmax(), reward)
        
        # Update agent (simplified)
        if len(memory) > 32:
            states, actions, rewards = memory.sample(32)
            # Policy gradient update...

torch.save(gating.state_dict(), 'models/gating_network.pth')
torch.save(agent.state_dict(), 'models/rl_agent.pth')
print("✅ Gating + RL Agent training done!")
```

---

# PHẦN 3: DEMO CHẠY SIGNAL

```python
# Import và chạy signal generation
import sys
sys.path.insert(0, '/content/drive/MyDrive/AI_SWING')

from run_signals import generate_signal

# Generate signal
signal = generate_signal(
    symbol='BTCUSDT',
    timeframe='1h',
    use_trained_models=True  # Dùng models đã train
)

print(f"📊 Signal: {signal['signal']}")
print(f"🎯 Entry: {signal['entry']}")
print(f"🛡️ SL: {signal['stop_loss']}")
print(f"🎯 TP: {signal['take_profit']}")
```

---

## Mẹo Tối Ưu Cho Colab

| Vấn Đề | Giải Pháp |
|--------|-----------|
| RAM không đủ | Giảm `batch_size` xuống 32 |
| GPU không dùng được | Kiểm tra: `torch.cuda.is_available()` |
| Runtime bị disconnect | Lưu checkpoint mỗi 10 epochs |
| Dữ liệu lớn | Train theo từng chunk |

## Lưu Ý Quan Trọng

1. **Backup thường xuyên** - Lưu model lên Google Drive
2. **API Keys** - Chỉ cần Binance API (ko cần Telegram)
3. **Dữ liệu tối thiểu** - Nên có ít nhất 1000 candles
4. **Thứ tự train** → Experts trước → Gating → RL Agent

## Thời Gian Training Ước Tính

- MLP Expert: ~10-15 phút
- CNN Expert: ~15-20 phút
- GRU Expert: ~20-30 phút
- Gating + RL: ~30-45 phút

**Tổng cộng: ~1.5 - 2 giờ**

---
**Chúc bạn training thành công! 🚀**