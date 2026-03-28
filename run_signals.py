# run_signals.py
import time
import schedule
import yaml
import logging
import concurrent.futures
from typing import Dict, Any, List
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datetime import datetime

from frontend.telegram_bot import TelegramBot
from pipeline.fetchers.binance_futures import BinanceFuturesFetcher
from pipeline.indicators.volatility import calculate_atr
from ai_core.signal_parser import SignalParser
from ai_core.rl_agent import PPOAgent

# Global config variable
CONFIG: Dict[str, Any] = {}

def setup_logging() -> None:
    """Configures the logging system."""
    log_level = CONFIG.get('system', {}).get('log_level', 'INFO')
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler("scanner.log", mode='a'),
            logging.StreamHandler()
        ]
    )

def load_ai_model() -> Any:
    """Loads the trained AI model."""
    weights_path = CONFIG.get('model', {}).get('weights_path', '')
    if not weights_path or not Path(weights_path).exists():
        logging.warning("AI model weights not found. Using random action.")
        return None
    
    state_dim = 29 # As determined from the build_data.py script output
    action_dim = 3
    model = PPOAgent(state_dim, action_dim, lr=0.0, gamma=0.0, K_epochs=0, eps_clip=0.0)
    try:
        model.policy.load_state_dict(torch.load(weights_path))
        model.policy.eval()
        logging.info(f"AI model loaded from {weights_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return None

def preprocess_for_inference(df: pd.DataFrame, scaler: Any, window_size: int = 60) -> torch.Tensor | None:
    """Prepares the most recent data for model inference."""
    if len(df) < window_size:
        logging.warning(f"Not enough data for inference window ({len(df)} < {window_size})")
        return None
    
    # Select feature columns (all numeric columns in this case)
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Get the last `window_size` rows
    recent_df = df[feature_cols].tail(window_size)
    
    # Scale the data
    scaled_data = scaler.transform(recent_df.values)
    
    # Convert to tensor and add batch dimension
    tensor = torch.FloatTensor(scaled_data).unsqueeze(0)
    return tensor

def process_symbol(symbol: str, timeframe: str, model: Any, bot: TelegramBot, scaler: Any) -> None:
    """
    Fetches data, gets an AI action, parses the signal, and sends it for a single symbol.
    """
    try:
        logging.info(f"--- Analyzing {symbol} on {timeframe} ---")
        fetcher = BinanceFuturesFetcher()
        df = fetcher.get_historical_data(symbol=symbol, interval=timeframe, start_str="10 days ago")
        
        if df is None or df.empty:
            logging.warning(f"No data for {symbol} on {timeframe}.")
            return

        # Add indicators
        df = calculate_atr(df, period=14)
        if df['atr'].isnull().all():
            logging.warning(f"Could not calculate ATR for {symbol}. Skipping.")
            return

        current_price = df['close'].iloc[-1]
        atr_value = df['atr'].iloc[-1]

        # Get AI action
        action = 1 # Default to HOLD
        if model and scaler:
            state_tensor = preprocess_for_inference(df, scaler)
            if state_tensor is not None:
                action, _ = model.select_action(state_tensor)
                logging.info(f"AI action for {symbol}: {action}")
        else:
            action = np.random.randint(0, 3)
            logging.info(f"Using random action for {symbol}: {action}")

        # Parse signal
        signal_parser = SignalParser()
        signal = signal_parser.parse_ai_action(action, current_price, atr_value)

        # Send signal
        if signal.get('signal') != "HOLD":
            logging.info(f"FOUND SIGNAL for {symbol}: {signal}")
            bot.send_signal(signal)

    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {e}", exc_info=True)

def scan_job(timeframe: str, model: Any, bot: TelegramBot, symbols: List[str], scaler: Any) -> None:
    """
    The main job function. It processes symbols concurrently.
    """
    logging.info(f"========== Running {timeframe} Scan Cycle ==========")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_symbol, symbol, timeframe, model, bot, scaler) for symbol in symbols]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"A symbol processing task failed: {e}")
    logging.info(f"========== {timeframe} Scan Cycle Finished ==========")

def main() -> None:
    """Sets up and runs the main scheduling loop."""
    global CONFIG
    try:
        with open("config/settings.yaml", 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        print("FATAL: config/settings.yaml not found. Please create it.")
        return
        
    setup_logging()
    
    bot = TelegramBot()
    bot.run_in_thread()
    
    ai_model = load_ai_model()
    
    # Load the scaler
    scaler = None
    try:
        with open("data/processed/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded successfully.")
    except FileNotFoundError:
        logging.error("Scaler file not found at data/processed/scaler.pkl. Inference will not be accurate.")

    symbols_to_scan = CONFIG.get('data', {}).get('symbols', [])
    
    logging.info("AI Signal Scanner started.")
    bot.send_message("🤖 AI Signal Scanner has started.")

    schedule.every().hour.at(":01").do(scan_job, timeframe='1h', model=ai_model, bot=bot, symbols=symbols_to_scan, scaler=scaler)
    for hour in ["00:01", "04:01", "08:01", "12:01", "16:01", "20:01"]:
        schedule.every().day.at(hour).do(scan_job, timeframe='4h', model=ai_model, bot=bot, symbols=symbols_to_scan, scaler=scaler)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
