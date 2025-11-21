import time
from datetime import datetime, timedelta
from typing import List, Optional
import csv, io

from config import HTTP_RETRIES
from schemas import (
    GDELTSearchIn, GDELTSearchOut, Article,
    PricesIn, PricesOut, PricePoint,
    EventReturnIn, EventReturnOut, EventReturn
)
from gdelt_client import gdelt_get
from stooq_client import stooq_download_csv


def gdelt_search(inp: GDELTSearchIn) -> GDELTSearchOut:
    params = {
        "query": inp.query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": inp.start_datetime,
        "enddatetime": inp.end_datetime,
        "maxrecords": inp.max_records,
        "sort": "HybridRel"
    }
    data = gdelt_get(params)
    arts = []

    for a in data.get("articles", []):
        seendate = (a.get("seendate", "") or "").strip()
        dt_iso = ""

        try:
            if seendate.isdigit():
                dt = datetime.strptime(seendate[:8], "%Y%m%d")
                dt_iso = dt.strftime("%Y-%m-%d")
            else:
                dt = datetime.fromisoformat(seendate.replace("Z", "").replace("T", " "))
                dt_iso = dt.strftime("%Y-%m-%d")
        except Exception:
            dt_iso = seendate[:10]

        arts.append(Article(
            title=a.get("title", ""),
            url=a.get("url", ""),
            datetime=dt_iso,
            source_country=a.get("sourceCountry"),
            language=a.get("language"),
            snippet=a.get("snippet")
        ))

    return GDELTSearchOut(articles=arts)


def gdelt_search_retry(inp: GDELTSearchIn) -> GDELTSearchOut:
    last = None
    for i in range(HTTP_RETRIES):
        try:
            return gdelt_search(inp)
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** i))
    raise last


def stooq_prices(inp: PricesIn) -> PricesOut:
    text = stooq_download_csv(inp.ticker, inp.start_date, inp.end_date)
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    prices = []

    for row in reader:
        prices.append(PricePoint(
            date=row["Date"],
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]) if row.get("Volume") else None
        ))

    return PricesOut(ticker=inp.ticker, prices=prices)


def compute_event_returns(inp: EventReturnIn) -> EventReturnOut:
    """
    Считаем доходность вокруг новости:
    pre_close — close за window_days ДО новости
    post_close — close за window_days ПОСЛЕ новости
    return_pct = (post / pre - 1) * 100
    """
    price_by_date = {p.date: p for p in inp.prices.prices}

    def nearest_close(d: datetime) -> Optional[float]:
        for k in range(0, 7):
            dd = (d - timedelta(days=k)).strftime("%Y-%m-%d")
            if dd in price_by_date:
                return price_by_date[dd].close
        return None

    def nearest_close_after(d: datetime) -> Optional[float]:
        for k in range(0, 7):
            dd = (d + timedelta(days=k)).strftime("%Y-%m-%d")
            if dd in price_by_date:
                return price_by_date[dd].close
        return None

    ers: List[EventReturn] = []
    for art in inp.articles:
        try:
            event_dt = datetime.fromisoformat(art.datetime)
        except Exception:
            continue

        pre_dt = event_dt - timedelta(days=inp.window_days)
        post_dt = event_dt + timedelta(days=inp.window_days)

        pre_close = nearest_close(pre_dt)
        post_close = nearest_close_after(post_dt)

        ret = None
        if pre_close and post_close:
            ret = (post_close / pre_close - 1) * 100

        ers.append(EventReturn(
            url=art.url,
            title=art.title,
            event_date=event_dt.strftime("%Y-%m-%d"),
            pre_close=pre_close,
            post_close=post_close,
            return_pct=round(ret, 3) if ret is not None else None
        ))

    return EventReturnOut(event_returns=ers)
