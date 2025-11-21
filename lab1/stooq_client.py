import requests, csv, io
from config import STOOQ_BASE, HTTP_TIMEOUT

def stooq_download_csv(ticker: str, start_date: str, end_date: str):
    # Stooq CSV download endpoint:
    # https://stooq.com/q/d/l/?s=AAPL.US&i=d
    params = {"s": ticker, "i": "d"}
    r = requests.get(STOOQ_BASE, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text
