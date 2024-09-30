import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("render_token_data_fetch.log"),  # Logs to a file
        logging.StreamHandler()  # Logs to console
    ]
)

# Retry decorator: Retry up to 3 times with a 2-second wait between attempts
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_binance_klines(symbol, interval, start_time, end_time):
    """
    Fetch klines data from Binance API for a given symbol and time range.

    Parameters:
    - symbol (str): Trading pair symbol (e.g., 'RNDRUSDT').
    - interval (str): Kline interval (e.g., '1d' for daily).
    - start_time (int): Start time in Unix timestamp (milliseconds).
    - end_time (int): End time in Unix timestamp (milliseconds).

    Returns:
    - list: JSON response from Binance API containing klines data.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_binance_price(symbol):
    """
    Fetch the latest price for a given symbol from Binance.

    Parameters:
    - symbol (str): Trading pair symbol (e.g., 'USDTBRL').

    Returns:
    - float: Latest price of the symbol.
    """
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return float(response.json()['price'])

def fetch_render_token_data(
    start_date: str,
    end_date: str,
    interval: str = "1h",
    interval_days: int = 7,
    output_path: str = "../../data/raw/crypto_prices/"
):
    """
    Fetch historical Render Token data from Binance in specified intervals.

    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - interval (str): Kline interval (default '1d' for daily data).
    - interval_days (int): Number of days per API request chunk.
    - output_path (str): Path to save the CSV file.

    Returns:
    - pandas.DataFrame: Aggregated data of Render Token prices in BRL.
    """
    all_data = []

    # Convert string dates to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Set the trading pair symbol to Render Token against USDT
    symbol = "RNDRUSDT"

    # Symbol for USDT against BRL
    usdt_brl_symbol = "USDTBRL"

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Fetch the latest USDTBRL rate
    try:
        usdt_brl_rate = fetch_binance_price(usdt_brl_symbol)
        logging.info(f"Fetched USDTBRL rate: {usdt_brl_rate}")
    except Exception as e:
        logging.error(f"Failed to fetch USDTBRL rate: {e}")
        return pd.DataFrame()

    current_start = start_dt
    while current_start < end_dt:
        current_end = current_start + timedelta(days=interval_days)
        if current_end > end_dt:
            current_end = end_dt

        # Convert to Unix timestamp in milliseconds
        start_timestamp = int(current_start.timestamp() * 1000)
        end_timestamp = int(current_end.timestamp() * 1000)

        logging.info(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")

        try:
            data = fetch_binance_klines(symbol, interval, start_timestamp, end_timestamp)
            if data:
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ])
                df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
                df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
                df["close"] = df["close"].astype(float)
                df.rename(columns={"close": "price_usdt"}, inplace=True)
                
                # Convert price from USDT to BRL
                df["price_brl"] = df["price_usdt"] * usdt_brl_rate
                all_data.append(df)
                logging.info(f"Fetched {len(df)} records for this interval.")
            else:
                logging.warning(f"No data returned for {symbol} from {current_start.date()} to {current_end.date()}")
        except Exception as e:
            logging.error(f"Failed to fetch data for {symbol} from {current_start.date()} to {current_end.date()}: {e}")

        # Move to the next interval, adding 1 millisecond to avoid overlap
        current_start = current_end + timedelta(milliseconds=1)

    if all_data:
        full_df = pd.concat(all_data).reset_index(drop=True)
        csv_file = os.path.join(output_path, "render_token_weekly_brl.csv")
        try:
            full_df.to_csv(csv_file, index=False)
            logging.info(f"Data successfully saved to {csv_file}")
        except Exception as e:
            logging.error(f"Failed to save data to CSV: {e}")
        return full_df
    else:
        logging.error("No data was fetched. Exiting without creating CSV.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage for Render Token
    start = "2023-09-29"
    end = "2024-09-29"
    render_token_data = fetch_render_token_data(start, end)
