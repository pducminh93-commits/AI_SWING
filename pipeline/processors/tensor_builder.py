# pipeline/processors/tensor_builder.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from utils.logger import default_logger as logger

class TensorBuilder:
    """
    Xây dựng tensor từ DataFrame đã có chỉ báo.
    Hỗ trợ chuẩn hóa, tạo sequence, chia train/val/test.
    """
    def __init__(self, window_size=60, horizon=1, scaler_type='standard', feature_cols=None):
        """
        Args:
            window_size: số bước thời gian trong một sequence (lookback)
            horizon: số bước dự đoán (target shift)
            scaler_type: 'standard' (StandardScaler) hoặc 'minmax' (MinMaxScaler)
            feature_cols: danh sách tên cột sẽ dùng làm features (nếu None thì dùng tất cả cột số)
        """
        self.window_size = window_size
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.feature_cols = feature_cols
        self.scaler = None

    def fit(self, df):
        """Khởi tạo scaler dựa trên toàn bộ dữ liệu."""
        if self.feature_cols is None:
            # Tự động chọn các cột số (loại float/int)
            self.feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Tự động chọn {len(self.feature_cols)} cột số làm features")
        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        self.scaler.fit(df[self.feature_cols].values)
        return self

    def transform(self, df):
        """Chuẩn hóa dữ liệu và tạo sequences."""
        # Chuẩn hóa features
        scaled = self.scaler.transform(df[self.feature_cols].values)
        # Tạo sequences
        X, y = [], []
        for i in range(len(scaled) - self.window_size - self.horizon + 1):
            X.append(scaled[i:i+self.window_size])
            # Target: giá đóng cửa tại step future (có thể dùng close hoặc returns)
            # Mặc định target là close price (cột 'close' trong feature_cols)
            close_idx = self.feature_cols.index('close') if 'close' in self.feature_cols else -1
            if close_idx != -1:
                # Target là giá đóng cửa sau horizon bước
                y.append(scaled[i+self.window_size+self.horizon-1, close_idx])
            else:
                # Fallback: target là close price thực tế chưa scale? Không nên.
                # Có thể dùng returns từ giá thực, nhưng tạm thời lấy close từ df gốc
                # Đơn giản: dùng close thực (không scale) để tính loss
                close_actual = df['close'].values
                y.append(close_actual[i+self.window_size+self.horizon-1])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def fit_transform(self, df):
        """Fit scaler và tạo sequences."""
        self.fit(df)
        return self.transform(df)

    def save_scaler(self, path):
        """Lưu scaler đã fit để dùng sau."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {path}")

    def load_scaler(self, path):
        """Load scaler đã lưu."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {path}")

def create_tensor_dataset(symbol, processed_dir, tensor_dir, window_size=60, horizon=1, val_ratio=0.2, test_ratio=0.1):
    """
    Đọc file parquet đã xử lý cho symbol, xây dựng tensor và chia tập.
    Lưu các file .npy và scaler.pkl.
    """
    processed_dir = Path(processed_dir)
    tensor_dir = Path(tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu
    file_path = processed_dir / f"{symbol}_features.parquet"
    if not file_path.exists():
        logger.error(f"File {file_path} không tồn tại!")
        return None
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {symbol}: shape {df.shape}")

    # Khởi tạo builder
    builder = TensorBuilder(window_size=window_size, horizon=horizon)

    # Fit scaler và transform
    X, y = builder.fit_transform(df)

    # Chia train/val/test theo thời gian (không xáo trộn)
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # Lưu
    np.save(tensor_dir / f"{symbol}_X_train.npy", X_train)
    np.save(tensor_dir / f"{symbol}_y_train.npy", y_train)
    np.save(tensor_dir / f"{symbol}_X_val.npy", X_val)
    np.save(tensor_dir / f"{symbol}_y_val.npy", y_val)
    np.save(tensor_dir / f"{symbol}_X_test.npy", X_test)
    np.save(tensor_dir / f"{symbol}_y_test.npy", y_test)

    builder.save_scaler(tensor_dir / f"{symbol}_scaler.pkl")

    logger.info(f"Saved tensors for {symbol}: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    return builder

if __name__ == "__main__":
    # Test
    import yaml
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    symbols = config['data']['symbols']
    processed_dir = Path("data/processed")
    tensor_dir = Path("data/tensors")
    for symbol in symbols:
        create_tensor_dataset(symbol, processed_dir, tensor_dir, window_size=60, horizon=1)