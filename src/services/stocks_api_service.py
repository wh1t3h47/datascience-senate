from typing import Union, Literal
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from os import getenv
from pandas import DataFrame
from yfinance import Ticker
from datetime import date, timedelta
from re import compile

from ..models.TransactionModel import ErrorEnum

load_dotenv()

is_url = compile("\/q\?s=(.*?)\"")

PeriodType = Union[Literal['1d'], Literal['7d'], Literal['15d'], Literal['1m'], Literal['6m'], Literal['1y']]

def extract_stock_symbol(url):
    match = is_url.search(url)
    if match:
        return match.group(1)
    else:
        return None

def get_symbol(keyword):
    """Retrieves the first matching stock symbol from Yahoo Finance."""
    print('------____>>', keyword)
    ticker = Ticker(keyword)
    return ticker.info.get("symbol")

def calculate_end_date(date: date, period: PeriodType) -> date:
    period_mapping: dict[PeriodType, int] = {
        "1d": 1,
        "7d": 7,
        "15d": 15,
        "1m": 30,
        "6m": 180,
        "1y": 365,
    }

    days = period_mapping[period]
    end_date = date + timedelta(days=days)
    return end_date

def stocks_api_service(symbol, date: date, period: PeriodType, ticker=""):
    """Fetches the average price of a stock on a specified date."""
    if ticker and ticker != '--':
        symbol = extract_stock_symbol(ticker)
    else:
        return None
        # @todo else search tick
    print('--->', symbol)

    _ticker = Ticker(symbol)
    
    # Change this line to dynamically calculate the end date based on the period
    end_date = calculate_end_date(date, period)
    
    hist: DataFrame = None
    hist_set = False
    try:
        hist = _ticker.history(start=date, end=end_date, raise_errors=True)
        hist_set = True
    except Exception as err:
        msg = str(err)
        if "symbol may be delisted" in msg:
            print('DELISTED')
            return ErrorEnum['SYMBOL_DELISTED']
        # else
        if "Data doesn't exist for" in msg:
            print("NOT FOUND")
            return ErrorEnum["DATA_NOT_FOUND"]
    
    # Verifique se há dados disponíveis para a data especificada
    if hist_set and not hist.empty:
        # Calcule a média dos preços de abertura
        average_price = hist["Open"].mean()
        print("Average Open Price ->", average_price, "\n")
        return average_price
    else:
        print(f"DATA NOT FOUND")
        return ErrorEnum["DATA_NOT_FOUND"]
