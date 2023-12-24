from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from os import getenv
from yfinance import Ticker
from datetime import date
from re import compile

load_dotenv()

is_url = compile("\/q\?s=(.*?)\"")
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

def stocks_api_service(symbol, date: date, ticker=""):
    """Fetches the average price of a stock on a specified date."""
    #print('------____>>', symbol, '| ', date)
    """
    symbol = "AAPL"
    
    if not symbol:
        print(f"Symbol not found for {name}")
        return None
    """
    if ticker and ticker != '--':
        symbol=extract_stock_symbol(ticker)
    else:
        return None
        # @todo else search tick
    print('--->', symbol)

    _ticker = Ticker(symbol)
    hist = _ticker.history(period="1d", start=date)
    
    # Verifique se há dados disponíveis para a data especificada
    if not hist.empty:
        # Calcule a média dos preços de abertura
        average_price = hist["Open"].mean()
        print("Average Open Price ->", average_price, "\n")
        return average_price
    else:
        print(f"No data found for {symbol} on {date}")
        return None