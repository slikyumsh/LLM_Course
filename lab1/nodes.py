import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI

from config import BASE_URL, API_KEY, MODEL_NAME, LLM_TEMPERATURE, MAX_ARTICLES
from schemas import (
    GraphState, PlanSpec, FinalReport,
    GDELTSearchIn, PricesIn, EventReturnIn,
    SentimentImpact, ImpactSummary,
)
from prompts import planner_prompt, sentiment_prompt, impact_prompt, reviewer_prompt
from tools import gdelt_search_retry, stooq_prices, compute_event_returns
from llm_utils import invoke_and_parse

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=LLM_TEMPERATURE,
    max_retries=0,
    timeout=120,
)

def _sj(obj):
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, ensure_ascii=False, indent=2)

def planner_view(state: GraphState) -> str:
    req = state.user_request.model_dump() if state.user_request else None
    return json.dumps(
        {
            "user_request": req,
            "have_articles": bool(state.articles),
            "articles_count": len(state.articles),
            "articles_sample_titles": [a.title for a in state.articles[:3]],
            "have_prices": state.prices is not None,
            "prices_points": len(state.prices.prices) if state.prices else 0,
            "have_event_returns": state.event_returns is not None,
            "event_returns_count": len(state.event_returns.event_returns)
            if state.event_returns else 0,
        },
        ensure_ascii=False,
        indent=2,
    )

def impact_view(state: GraphState) -> str:
    req = state.user_request
    return json.dumps(
        {
            "ticker": req.ticker if req else None,
            "company_name": req.company_name if req else None,
            "sentiments": [s.model_dump() for s in state.sentiments],
            "event_returns": (
                [er.model_dump() for er in state.event_returns.event_returns]
                if state.event_returns else []
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

def writer_view(state: GraphState) -> str:
    req = state.user_request
    return json.dumps(
        {
            "ticker": req.ticker if req else None,
            "company": req.company_name if req else None,
            "lookback_days": req.lookback_days if req else None,
            "event_window_days": req.event_window_days if req else None,
            "articles_analyzed": len(state.articles),
            "impact_summary": state.impact_summary.model_dump()
            if state.impact_summary else None,
            "event_returns": (
                [er.model_dump() for er in state.event_returns.event_returns]
                if state.event_returns else []
            ),
            "example_headlines": [a.title for a in state.articles[:5]],
        },
        ensure_ascii=False,
        indent=2,
    )

def planner_node(state: GraphState) -> GraphState:
    prompt = planner_prompt().format(state_json=planner_view(state))
    plan = invoke_and_parse(llm, PlanSpec, prompt)
    state.plan = plan
    return state

def sentiment_agent_map_node(state: GraphState) -> GraphState:
    """
    Параллельный LLM-map по новостям.
    """
    def one_article(art):
        d = art.model_dump()

        # слишком длинный сниппет
        if d.get("snippet") and len(d["snippet"]) > 800:
            d["snippet"] = d["snippet"][:800] + "..."

        p = sentiment_prompt().format(article_json=_sj(d))
        s = invoke_and_parse(llm, SentimentImpact, p)

        if not s.url:
            s.url = d.get("url")
        if not s.title:
            s.title = d.get("title")

        return s

    sentiments: list[SentimentImpact] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(one_article, art) for art in state.articles[:MAX_ARTICLES]]
        for f in as_completed(futs):
            sentiments.append(f.result())

    state.sentiments = sentiments
    return state

def impact_estimator_node(state: GraphState) -> GraphState:
    prompt = impact_prompt().format(state_json=impact_view(state))
    summary = invoke_and_parse(llm, ImpactSummary, prompt)

    if not summary.per_article:
        summary.per_article = state.sentiments

    if not summary.strongest_positive:
        # по polarity*confidence 
        pos = sorted(
            [s for s in state.sentiments if s.sentiment == "positive"],
            key=lambda x: (x.polarity * x.confidence),
            reverse=True
        )
        summary.strongest_positive = [s.url for s in pos[:3] if s.url]

    if not summary.strongest_negative:
        neg = sorted(
            [s for s in state.sentiments if s.sentiment == "negative"],
            key=lambda x: (abs(x.polarity) * x.confidence),
            reverse=True
        )
        summary.strongest_negative = [s.url for s in neg[:3] if s.url]

    if not summary.overall_assessment:
        n_pos = sum(1 for s in state.sentiments if s.sentiment == "positive")
        n_neg = sum(1 for s in state.sentiments if s.sentiment == "negative")
        avg_pol = (
            sum(s.polarity for s in state.sentiments) / max(1, len(state.sentiments))
        )
        summary.overall_assessment = (
            f"За выбранный период найдено {len(state.sentiments)} новостей: "
            f"{n_pos} позитивных, {n_neg} негативных. "
            f"Средняя тональность {avg_pol:.2f}. "
            "Связь с краткосрочной доходностью оценена через событийнyю доходность; "
            "часть новостей показывает совпадение ожидаемого и фактического направления."
        )

    state.impact_summary = summary
    return state


def reviewer_writer_node(state: GraphState) -> GraphState:
    prompt = reviewer_prompt().format(state_json=writer_view(state))
    report = invoke_and_parse(llm, FinalReport, prompt)

    req = state.user_request

    if not report.window:
        lb = req.lookback_days if req else "?"
        ew = req.event_window_days if req else "?"
        report.window = f"lookback {lb}d, event ±{ew}d"

    if not report.conclusion:
        overall = (
            state.impact_summary.overall_assessment
            if state.impact_summary and state.impact_summary.overall_assessment
            else ""
        )
        n_pos = sum(1 for s in state.sentiments if s.sentiment == "positive")
        n_neg = sum(1 for s in state.sentiments if s.sentiment == "negative")
        report.conclusion = (
            f"Анализ выполнен на основе {len(state.articles)} новостей: "
            f"{n_pos} позитивных и {n_neg} негативных по тону. "
            f"{overall} "
            "Событийная доходность использовалась как учебная оценка краткосрочной реакции рынка. "
            "Результаты носят исследовательский характер и не являются инвестиционной рекомендацией."
        )

    state.report = report
    return state


def gdelt_search_node(state: GraphState) -> GraphState:
    req = state.plan.normalized_request
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=req.lookback_days)

    out = gdelt_search_retry(
        GDELTSearchIn(
            query=req.company_name,
            start_datetime=start_dt.strftime("%Y%m%d%H%M%S"),
            end_datetime=end_dt.strftime("%Y%m%d%H%M%S"),
            max_records=req.max_articles,
        )
    )
    state.articles = out.articles
    return state

def stooq_prices_node(state: GraphState) -> GraphState:
    req = state.plan.normalized_request
    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=req.lookback_days + 10)

    out = stooq_prices(
        PricesIn(
            ticker=req.ticker,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
    )
    state.prices = out
    return state

def event_returns_node(state: GraphState) -> GraphState:
    req = state.plan.normalized_request
    out = compute_event_returns(
        EventReturnIn(
            prices=state.prices,
            articles=state.articles,
            window_days=req.event_window_days,
        )
    )
    state.event_returns = out
    return state
