import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from pipeline.fetchers.binance_futures import fetch_all_data
from pipeline.processors.timeframe_sync import sync_timeframes_for_symbol
from pipeline.indicators import trend, momentum, volatility, order_flow
from utils.logger import logger

def add_time_features(df):
    """
    Thêm mắt thần thời gian: Giúp AI biết chu kỳ phiên Á, Âu, Mỹ
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    # Sin-Cos Encoding để AI hiểu tính chu kỳ (23h gần với 0h)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def main():
    # 0. Load Configuration
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    symbols = config['data']['symbols']
    timeframes = config['data']['timeframes']
    lookback = config['data']['train_lookback_days']

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch raw data for all symbols
    for symbol in symbols:
        logger.info(f"🚀 --- Fetching data for {symbol} ---")
        try:
            fetch_all_data(symbol, timeframes, lookback, raw_dir)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    all_symbols_df = []

    # 2. Process each symbol
    for symbol in symbols:
        logger.info(f"🛠️ --- Processing {symbol} ---")
        data_frames = {}
        for tf in timeframes:
            file_path = raw_dir / f"{symbol}_{tf}.csv"
            if file_path.exists():
                # Đảm bảo index là datetime để resample/join chuẩn
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                data_frames[tf] = df
            else:
                logger.warning(f"⚠️ Missing file {file_path}")
        
        if not data_frames:
            continue

        # Đồng bộ các khung giờ (Multi-timeframe sync)
        synced_df = sync_timeframes_for_symbol(symbol, data_frames)
        if synced_df is None:
            continue

        # Join dữ liệu Open Interest và Funding Rate (Nếu có)
        oi_path = raw_dir / f"{symbol}_oi.csv"
        funding_path = raw_dir / f"{symbol}_funding.csv"
        
        if oi_path.exists():
            oi_df = pd.read_csv(oi_path, index_col='timestamp', parse_dates=True)
            # Resample về 1h và điền khuyết dữ liệu (ffill)
            synced_df = synced_df.join(oi_df.resample('1h').last().ffill(), how='left')
        
        if funding_path.exists():
            funding_df = pd.read_csv(funding_path, index_col='timestamp', parse_dates=True)
            synced_df = synced_df.join(funding_df.resample('1h').last().ffill(), how='left')

        # 3. LẮP MẮT THẦN (INDICATORS)
        logger.info(f"👁️ Calculating Indicators for {symbol}...")
        
        # Trend (Xu hướng)
        synced_df = trend.calculate_ema(synced_df, periods=[20, 50, 200]) # Thêm EMA 200 cho Swing
        synced_df = trend.calculate_macd(synced_df)
        
        # Momentum (Động lượng)
        synced_df = momentum.calculate_rsi(synced_df, period=14)
        synced_df = momentum.calculate_stochastic(synced_df)
        
        # Volatility (Biến động)
        synced_df = volatility.calculate_atr(synced_df, period=14)
        # Thêm Bollinger Bands nếu file volatility.py có hỗ trợ
        if hasattr(volatility, 'calculate_bollinger_bands'):
            synced_df = volatility.calculate_bollinger_bands(synced_df)
        
        # Time Features (Thời gian)
        synced_df = add_time_features(synced_df)

        # Order Flow Divergence (Nếu có data OI)
        if 'open_interest' in synced_df.columns and hasattr(order_flow, 'add_oi_divergence'):
            synced_df = order_flow.add_oi_divergence(synced_df)
        
        synced_df['symbol_name'] = symbol # Đổi tên tránh trùng hàm hệ thống
        all_symbols_df.append(synced_df)

    # 4. Combine & Save
    if all_symbols_df:
        final_df = pd.concat(all_symbols_df)
        
        # Xử lý dữ liệu khuyết sau khi tính Indicators (Thường là các dòng đầu)
        before_count = len(final_df)
        final_df.dropna(inplace=True)
        after_count = len(final_df)
        
        out_path = processed_dir / "data_futures_swing.parquet"
        final_df.to_parquet(out_path)
        
        logger.info(f"✅ HOÀN TẤT: {out_path}")
        logger.info(f"📊 Kích thước: {final_df.shape} (Đã loại bỏ {before_count - after_count} dòng NaN)")
    else:
        logger.error("❌ Không có dữ liệu nào được xử lý thành công.")

if __name__ == "__main__":
    main()
