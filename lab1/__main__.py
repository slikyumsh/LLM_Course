import json
from pathlib import Path
from schemas import GraphState, UserRequest
from graph import build_graph

def save_graph_diagrams(app, out_dir="diagrams"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = app.get_graph(xray=True)

    mermaid_str = g.draw_mermaid()
    (out_dir / "graph.mmd").write_text(mermaid_str, encoding="utf-8")
    print(f"[mmd scheme] Mermaid saved: {out_dir/'graph.mmd'}")

    png_bytes = g.draw_mermaid_png()
    (out_dir / "graph.png").write_bytes(png_bytes)
    print(f"[png scheme] PNG saved: {out_dir/'graph.png'}")
   

def demo():
    app = build_graph()
    save_graph_diagrams(app)

    init_state = GraphState(
        user_request=UserRequest(
            ticker="AMZN.US",
            company_name="Amazon",
            lookback_days=3,
            event_window_days=1,
            max_articles=3
        )
    )

    out = app.invoke(init_state)

    print("!!! Final report !!!")
    report = out.get("report")
    if hasattr(report, "model_dump_json"):
        print(report.model_dump_json(indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo()