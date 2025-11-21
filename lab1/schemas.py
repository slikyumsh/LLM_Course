from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Any

class UserRequest(BaseModel):
    ticker: str
    company_name: str
    lookback_days: int = 7
    event_window_days: int = 1
    max_articles: int = 30


class ToolCall(BaseModel):
    tool_name: Literal[
        "gdelt_search",
        "stooq_prices",
        "compute_event_returns",
        "none"
    ]
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class PlanSpec(BaseModel):
    normalized_request: UserRequest
    strategy: str
    next_call: ToolCall


class GDELTSearchIn(BaseModel):
    query: str
    start_datetime: str
    end_datetime: str
    max_records: int = 30


class Article(BaseModel):
    title: str
    url: str
    datetime: str
    source_country: Optional[str] = None
    language: Optional[str] = None
    snippet: Optional[str] = None


class GDELTSearchOut(BaseModel):
    articles: List[Article]


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class PricesIn(BaseModel):
    ticker: str
    start_date: str
    end_date: str


class PricesOut(BaseModel):
    ticker: str
    prices: List[PricePoint]


class EventReturnIn(BaseModel):
    prices: PricesOut
    articles: List[Article]
    window_days: int


class EventReturn(BaseModel):
    url: str
    title: str
    event_date: str
    pre_close: Optional[float]
    post_close: Optional[float]
    return_pct: Optional[float]


class EventReturnOut(BaseModel):
    event_returns: List[EventReturn]


# outputs 
class SentimentImpact(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None

    sentiment: Literal["negative", "neutral", "positive"]
    polarity: float  # [-1..1]
    expected_impact: Literal["down", "neutral", "up"]
    confidence: float  # 0..1
    rationale: str


class ImpactSummary(BaseModel):
    per_article: List[SentimentImpact] = []
    strongest_positive: List[str] = []
    strongest_negative: List[str] = []
    overall_assessment: str = ""



class FinalReport(BaseModel):
    ticker: str
    company: str
    window: str = ""
    articles_analyzed: int
    impact_summary: ImpactSummary
    event_returns: List[EventReturn]
    conclusion: str = ""  


class GraphState(BaseModel):
    user_request: Optional[UserRequest] = None
    plan: Optional[PlanSpec] = None

    articles: List[Article] = []
    prices: Optional[PricesOut] = None
    event_returns: Optional[EventReturnOut] = None

    sentiments: List[SentimentImpact] = []
    impact_summary: Optional[ImpactSummary] = None
    report: Optional[FinalReport] = None
