# pipeline/processors/timeframe_sync.py
import pandas as pd
from pathlib import Path
from utils.logger import logger

def sync_timeframes_for_symbol(symbol, timeframes_dict):
    """
    Đồng bộ nhiều khung thời gian về khung nhỏ nhất.
    Args:
        symbol (str): Tên symbol (dùng để log)
        timeframes_dict (dict): key = timeframe (e.g., '1h', '4h', '1d'), value = DataFrame (index=timestamp)
    Returns:
        DataFrame đã đồng bộ (base timeframe + các cột từ khung lớn hơn với prefix), hoặc None nếu lỗi.
    """
    if not timeframes_dict:
        logger.error(f"{symbol}: Không có dữ liệu đầu vào.")
        return None

    # Xác định khung thời gian nhỏ nhất (dựa trên pd.Timedelta)
    try:
        sorted_tfs = sorted(timeframes_dict.keys(), key=lambda x: pd.Timedelta(x))
    except Exception as e:
        logger.error(f"{symbol}: Lỗi sắp xếp timeframes: {e}")
        return None

    base_tf = sorted_tfs[0]
    base_df = timeframes_dict[base_tf].copy()
    logger.info(f"{symbol}: Khung cơ sở = {base_tf}")

    # Đổi tên cột cho các khung lớn hơn để tránh trùng
    for tf, df in timeframes_dict.items():
        if tf == base_tf:
            continue
        df_renamed = df.copy()
        df_renamed.rename(columns=lambda col: f"{col}_{tf}", inplace=True)
        timeframes_dict[tf] = df_renamed

    # Dịch timestamp của các khung lớn hơn về thời điểm đóng nến (để forward fill chính xác)
    for tf, df in timeframes_dict.items():
        if tf == base_tf:
            continue
        delta = pd.Timedelta(tf)
        df.index = df.index + delta

    # Join các khung lớn hơn vào base_df
    synced_df = base_df.copy()
    for tf, df in timeframes_dict.items():
        if tf == base_tf:
            continue
        synced_df = synced_df.join(df, how='left')

    # Forward fill các giá trị từ khung lớn hơn xuống
    synced_df.ffill(inplace=True)

    # Xóa các dòng đầu tiên có NaN (khi chưa có dữ liệu từ khung lớn hơn)
    initial_len = len(synced_df)
    synced_df.dropna(inplace=True)
    dropped = initial_len - len(synced_df)
    if dropped > 0:
        logger.info(f"{symbol}: Đã xóa {dropped} dòng đầu do thiếu dữ liệu từ khung lớn.")

    logger.info(f"{symbol}: Đồng bộ xong, shape {synced_df.shape}")
    return synced_df

def load_and_sync(symbol, raw_dir, timeframes):
    """
    Tiện ích: Đọc các file CSV từ raw_dir cho các timeframes, đồng bộ và trả về DataFrame.
    Args:
        symbol (str): Tên symbol
        raw_dir (Path): Đường dẫn đến thư mục raw (chứa các file {symbol}_{tf}.csv)
        timeframes (list): Danh sách các timeframe cần đồng bộ
    Returns:
        DataFrame đã đồng bộ, hoặc None nếu lỗi.
    """
    raw_dir = Path(raw_dir)
    timeframes_dict = {}
    for tf in timeframes:
        file_path = raw_dir / f"{symbol}_{tf}.csv"
        if not file_path.exists():
            logger.error(f"{symbol}: File {file_path} không tồn tại.")
            return None
        try:
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            if df.empty:
                logger.warning(f"{symbol}: File {file_path} rỗng.")
                continue
            timeframes_dict[tf] = df
        except Exception as e:
            logger.error(f"{symbol}: Lỗi đọc file {file_path}: {e}")
            return None

    if not timeframes_dict:
        logger.error(f"{symbol}: Không có dữ liệu hợp lệ.")
        return None

    return sync_timeframes_for_symbol(symbol, timeframes_dict)

if __name__ == "__main__":
    # Test nhanh nếu chạy trực tiếp
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    symbols = config['data']['symbols']
    timeframes = config['data']['timeframes']
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        df_synced = load_and_sync(symbol, raw_dir, timeframes)
        if df_synced is not None:
            out_path = processed_dir / f"{symbol}_synced.parquet"
            df_synced.to_parquet(out_path)
            print(f"✅ {symbol}: Đã lưu {out_path} (shape {df_synced.shape})")