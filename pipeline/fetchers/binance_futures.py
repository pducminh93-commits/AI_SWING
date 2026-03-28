# pipeline/fetchers/binance_futures.py
import time
import pandas as pd
from pathlib import Path
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests.exceptions
from utils.logger import logger


class BinanceFuturesFetcher:
    def __init__(self, testnet=False):
        self.client = Client()
        if testnet:
            self.client.API_URL = 'https://testnet.binancefuture.com'

    def get_historical_data(self, symbol, interval, start_str):
        """
        Fetch historical klines for a symbol and interval.
        :param symbol: e.g., 'BTCUSDT'
        :param interval: e.g., '1h', '4h'
        :param start_str: e.g., '100 days ago'
        :return: DataFrame with columns: open, high, low, close, volume, buy_vol
        """
        try:
            klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )
            if not klines:
                logger.warning(f"⚠️ Không có dữ liệu cho {symbol} {interval}")
                return None

            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)

            # Ép kiểu số
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Giữ lại cột cần thiết và đổi tên
            df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']]
            df.rename(columns={'taker_buy_base_asset_volume': 'buy_vol'}, inplace=True)
            df.dropna(inplace=True)

            logger.info(f"✅ Đã lấy {len(df)} nến cho {symbol} {interval}")
            return df
        except Exception as e:
            logger.error(f"❌ Lỗi khi lấy dữ liệu cho {symbol} {interval}: {e}")
            return None

def fetch_futures_klines(symbol, interval, lookback_days, client=None):
    """
    Kéo nến USDⓈ-M Futures.
    Trả về DataFrame với các cột:
        open, high, low, close, volume, buy_vol (taker buy base asset volume)
    """
    if client is None:
        client = Client()
    
    logger.info(f"⏳ Đang kéo {symbol} - {interval} ({lookback_days} ngày)...")
    try:
        start_str = f"{lookback_days} days ago UTC"
        klines = client.futures_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str
        )
        if not klines:
            logger.warning(f"⚠️ Không có dữ liệu cho {symbol} {interval}")
            return None

        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(klines, columns=columns)

        # Ép kiểu số
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Giữ lại cột cần thiết và đổi tên
        df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']]
        df.rename(columns={'taker_buy_base_asset_volume': 'buy_vol'}, inplace=True)
        df.dropna(inplace=True)

        logger.info(f"✅ Đã lấy {len(df)} nến cho {symbol} {interval}")
        return df

    except BinanceAPIException as e:
        logger.error(f"❌ Lỗi API Binance ({symbol}): {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Lỗi kết nối mạng: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Lỗi hệ thống: {e}")
        return None


def fetch_open_interest(symbol, lookback_days, client=None):
    """
    Kéo lịch sử Open Interest của symbol.
    Endpoint: /futures/data/openInterestHist
    Trả về DataFrame với index timestamp và cột 'open_interest'.
    """
    if client is None:
        client = Client()

    logger.info(f"⏳ Đang kéo Open Interest cho {symbol} ({lookback_days} ngày)...")
    try:
        # Giới hạn tối đa 500 ngày, nhưng API trả về nhiều
        start_str = f"{lookback_days} days ago UTC"
        oi_data = client.futures_open_interest_hist(
            symbol=symbol,
            period="5m",  # 5m là chu kỳ dữ liệu OI lịch sử
            start_str=start_str,
            limit=1000   # mỗi request tối đa 1000 điểm
        )
        # Lưu ý: API này trả về danh sách dict với timestamp (ms) và sumOpenInterest
        if not oi_data:
            logger.warning(f"⚠️ Không có dữ liệu OI cho {symbol}")
            return None

        df = pd.DataFrame(oi_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['sumOpenInterest']].rename(columns={'sumOpenInterest': 'open_interest'})
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
        df.dropna(inplace=True)
        logger.info(f"✅ Đã lấy {len(df)} điểm OI cho {symbol}")
        return df

    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy OI {symbol}: {e}")
        return None


def fetch_funding_rate(symbol, lookback_days, client=None):
    """
    Kéo lịch sử Funding Rate của symbol.
    Endpoint: /futures/data/fundingRate
    Trả về DataFrame với index timestamp và cột 'funding_rate'.
    """
    if client is None:
        client = Client()

    logger.info(f"⏳ Đang kéo Funding Rate cho {symbol} ({lookback_days} ngày)...")
    try:
        start_str = f"{lookback_days} days ago UTC"
        funding_data = client.futures_funding_rate(
            symbol=symbol,
            start_str=start_str,
            limit=1000
        )
        if not funding_data:
            logger.warning(f"⚠️ Không có dữ liệu funding rate cho {symbol}")
            return None

        df = pd.DataFrame(funding_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        df.dropna(inplace=True)
        logger.info(f"✅ Đã lấy {len(df)} điểm funding rate cho {symbol}")
        return df

    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy funding rate {symbol}: {e}")
        return None


def fetch_all_data(symbol, intervals, lookback_days, raw_dir):
    """
    Kéo tất cả dữ liệu: nến, OI, funding rate cho một symbol.
    Lưu từng loại vào file riêng trong raw_dir.
    """
    client = Client()
    for interval in intervals:
        df_klines = fetch_futures_klines(symbol, interval, lookback_days, client)
        if df_klines is not None:
            file_path = Path(raw_dir) / f"{symbol}_{interval}.csv"
            df_klines.to_csv(file_path, encoding='utf-8-sig')
            logger.info(f"✅ Lưu {file_path}")

    # Fetch and save Open Interest (chỉ 1 lần)
    df_oi = fetch_open_interest(symbol, lookback_days, client)
    if df_oi is not None:
        oi_path = Path(raw_dir) / f"{symbol}_oi.csv"
        df_oi.to_csv(oi_path, encoding='utf-8-sig')
        logger.info(f"✅ Lưu {oi_path}")

    # Fetch and save Funding Rate (chỉ 1 lần)
    df_funding = fetch_funding_rate(symbol, lookback_days, client)
    if df_funding is not None:
        funding_path = Path(raw_dir) / f"{symbol}_funding.csv"
        df_funding.to_csv(funding_path, encoding='utf-8-sig')
        logger.info(f"✅ Lưu {funding_path}")