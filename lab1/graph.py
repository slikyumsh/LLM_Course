from typing import Literal
from langgraph.graph import StateGraph, END
from schemas import GraphState
from nodes import (
    planner_node, gdelt_search_node, stooq_prices_node, event_returns_node,
    sentiment_agent_map_node, impact_estimator_node, reviewer_writer_node
)

ROUTE_MAP = {
    "gdelt_search": "gdelt_search",
    "stooq_prices": "stooq_prices",
    "event_returns": "event_returns",
    "sentiment_map": "sentiment_map",
}

def route_from_planner(state: GraphState) -> Literal[
    "gdelt_search", "stooq_prices", "event_returns", "sentiment_map"
]:
    tool = state.plan.next_call.tool_name

    have_articles = bool(state.articles)
    have_prices = state.prices is not None
    have_returns = state.event_returns is not None

    if tool == "gdelt_search" and have_articles:
        tool = "stooq_prices"
    if tool == "stooq_prices" and have_prices:
        tool = "compute_event_returns"
    if tool == "compute_event_returns" and have_returns:
        tool = "none"

    if tool == "gdelt_search":
        return "gdelt_search"
    if tool == "stooq_prices":
        return "stooq_prices"
    if tool == "compute_event_returns":
        return "event_returns"
    return "sentiment_map"


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("planner", planner_node)
    g.add_node("gdelt_search", gdelt_search_node)
    g.add_node("stooq_prices", stooq_prices_node)
    g.add_node("event_returns", event_returns_node)

    g.add_node("sentiment_map", sentiment_agent_map_node)
    g.add_node("impact", impact_estimator_node)
    g.add_node("writer", reviewer_writer_node)

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", route_from_planner, ROUTE_MAP)

    g.add_edge("gdelt_search", "planner")
    g.add_edge("stooq_prices", "planner")
    g.add_edge("event_returns", "planner")

    g.add_edge("sentiment_map", "impact")
    g.add_edge("impact", "writer")
    g.add_edge("writer", END)

    return g.compile()
