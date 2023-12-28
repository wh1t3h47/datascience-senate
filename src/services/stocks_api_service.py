from typing import Literal
from dotenv import load_dotenv
from pandas import DataFrame
from yfinance import Ticker
from datetime import date, timedelta
from re import compile, search

from ..models.ErrorEnumModel import ErrorEnumModel

load_dotenv()

is_url = compile("\/q\?s=(.*?)\"")

PeriodType = Literal['1d', '7d', '15d', '1m', '6m', '1y']
DateType = Literal["start", "end"]
DailyAvarage = Literal["day", "avg"]
period_mapping: dict[PeriodType, int] = {
    "1d": 1,
    "7d": 7,
    "15d": 15,
    "1m": 30,
    "6m": 180,
    "1y": 365,
}

def extract_stock_symbol(url):
    match = is_url.search(url)
    if match:
        return match.group(1)
    else:
        return None
    
def get_ordinal_suffix(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return suffix

def get_symbol(keyword):
    """Retrieves the first matching stock symbol from Yahoo Finance."""
    print('------____>>', keyword)
    ticker = Ticker(keyword)
    return ticker.info.get("symbol")

def calculate_date(date: date, period: PeriodType, date_type: DateType, type: DailyAvarage) -> date:
    days = period_mapping[period]
    if date_type == "start":
        return date - timedelta(days=days - 1) if type == "day" else date - timedelta(days=days)
    else:  # date_type == "end"
        return date + timedelta(days=days)

def stocks_api_service(symbol: str, date: date, period: PeriodType, ticker: str, type: DailyAvarage):
    """Fetches the daily or average price of a stock on a specified date."""
    if ticker and ticker != '--':
        symbol = extract_stock_symbol(ticker)
    else:
        return None

    _ticker = Ticker(symbol)

    start_date = calculate_date(date, period, "start", type)
    end_date = calculate_date(date, period, "end", type)

    hist: DataFrame = None
    hist_set = False
    try:
        hist = _ticker.history(start=start_date, end=end_date, raise_errors=True)
        hist_set = True
    except Exception as err:
        msg = str(err)
        if "symbol may be delisted" in msg:
            return ErrorEnumModel['SYMBOL_DELISTED']
        if "Data doesn't exist for" in msg:
            return ErrorEnumModel["DATA_NOT_FOUND"]

    if hist_set and not hist.empty:
        # print(period)
        # Calculate the average of daily prices or the average price
        num_days = period_mapping[period] if period in period_mapping else int(search(r'\d+', period).group())
        price_type = "Average For" if type == "day" else "Average Since Bought Until"
        
        if type == "day":
            # Calculate the average for the specified day
            prices = hist["Open"].tail(num_days)
        else:
            # Calculate the average since bought until the specified days
            if num_days <= len(hist):
                prices = hist["Open"].iloc[:num_days]
            else:
                prices = hist["Open"]

        average_price = prices.mean()

        ordinal_suffix = get_ordinal_suffix(num_days)

        print(f"📈 Stock --> ({symbol})")
        print(f"{price_type} {num_days}{ordinal_suffix} day")
        print(f"💰 ${average_price:.2f}\n")

        return average_price
    else:
        print(f"DATA NOT FOUND")
        return ErrorEnumModel["DATA_NOT_FOUND"]
