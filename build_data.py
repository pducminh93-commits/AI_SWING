# build_data.py
import yaml
import pandas as pd
from pathlib import Path
from pipeline.fetchers.binance_futures import fetch_all_data
from pipeline.processors.timeframe_sync import sync_timeframes_for_symbol
from pipeline.indicators import trend, momentum, volatility, order_flow
from utils.logger import logger

def main():
    config_path = Path(__file__).parent / "config" / "settings.yaml"
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
        logger.info(f"--- Fetching data for {symbol} ---")
        fetch_all_data(symbol, timeframes, lookback, raw_dir)

    all_symbols_df = []

    # 2. Process each symbol and collect into a list
    for symbol in symbols:
        logger.info(f"--- Processing {symbol} ---")
        data_frames = {}
        for tf in timeframes:
            file_path = raw_dir / f"{symbol}_{tf}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                data_frames[tf] = df
            else:
                logger.error(f"Missing file {file_path}")
                continue
        
        if not data_frames:
            continue

        synced_df = sync_timeframes_for_symbol(symbol, data_frames)
        if synced_df is None:
            continue

        oi_path = raw_dir / f"{symbol}_oi.csv"
        funding_path = raw_dir / f"{symbol}_funding.csv"
        if oi_path.exists():
            oi_df = pd.read_csv(oi_path, index_col='timestamp', parse_dates=True)
            oi_resampled = oi_df.resample('1h').last().ffill()
            synced_df = synced_df.join(oi_resampled, how='left')
        if funding_path.exists():
            funding_df = pd.read_csv(funding_path, index_col='timestamp', parse_dates=True)
            funding_resampled = funding_df.resample('1h').last().ffill()
            synced_df = synced_df.join(funding_resampled, how='left')

        # Add indicators
        synced_df = trend.calculate_ema(synced_df, periods=[20, 50])
        synced_df = trend.calculate_macd(synced_df)
        synced_df = momentum.calculate_rsi(synced_df, period=14)
        synced_df = momentum.calculate_stochastic(synced_df)
        synced_df = volatility.calculate_atr(synced_df, period=14)
        # Placeholder for order flow
        # if 'open_interest' in synced_df.columns:
        #     synced_df = order_flow.add_oi_divergence(synced_df)
        
        synced_df['symbol'] = symbol
        all_symbols_df.append(synced_df)

    # 3. Combine all symbols into a single DataFrame and save
    if all_symbols_df:
        final_df = pd.concat(all_symbols_df)
        final_df.dropna(inplace=True)
        out_path = processed_dir / "data_futures_swing.parquet"
        final_df.to_parquet(out_path)
        logger.info(f"✅ Combined data saved to {out_path} (shape {final_df.shape})")
    else:
        logger.error("No data was processed.")

if __name__ == "__main__":
    main()
