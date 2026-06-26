#!/usr/bin/env python3
"""GitHub Actions batch runner for TradingAgents-Astock.

Reads configuration from environment variables and runs analysis for all
tickers in ``STOCK_LIST`` (comma-separated). Each ticker's decision and
full report are saved under ``$TRADINGAGENTS_RESULTS_DIR/<ticker>/<date>/``.

Environment variables
---------------------
STOCK_LIST : str
    Comma-separated ticker codes (e.g. ``600519,000858,300750``).
    Default: ``600519`` (贵州茅台).
TRADE_DATE : str, optional
    Analysis date in YYYY-MM-DD. Defaults to today (Asia/Shanghai).
LLM_PROVIDER : str
    LLM provider key. Default: ``minimax``.
DEEP_THINK_LLM : str
    Deep thinking model id. Default: ``MiniMax-M2.7``.
QUICK_THINK_LLM : str
    Quick thinking model id. Default: ``MiniMax-M2.7-highspeed``.
BACKEND_URL : str, optional
    Custom API base URL for OpenAI-compatible providers.
MAX_DEBATE_ROUNDS : str
    Debate rounds. Default: ``1``.
MAX_RISK_DISCUSS_ROUNDS : str
    Risk discussion rounds. Default: ``1``.
OUTPUT_LANGUAGE : str
    Report language. Default: ``Chinese``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def main() -> int:
    # ── Configuration ────────────────────────────────────────────────────
    stock_list = _env("STOCK_LIST", "600519")
    trade_date = _env("TRADE_DATE") or datetime.now(
        timezone(timedelta(hours=8))
    ).strftime("%Y-%m-%d")

    llm_provider = _env("LLM_PROVIDER", "deepseek")
    deep_think_llm = _env("DEEP_THINK_LLM", "deepseek-v4-pro")
    quick_think_llm = _env("QUICK_THINK_LLM", "deepseek-v4-flash")
    backend_url = _env("BACKEND_URL")

    results_base = _env(
        "TRADINGAGENTS_RESULTS_DIR",
        os.path.join(os.path.expanduser("~"), ".tradingagents", "logs"),
    )

    max_debate_rounds = int(_env("MAX_DEBATE_ROUNDS", "1"))
    max_risk_discuss_rounds = int(_env("MAX_RISK_DISCUSS_ROUNDS", "1"))
    output_language = _env("OUTPUT_LANGUAGE", "Chinese")

    # Parse ticker list
    tickers = [t.strip() for t in stock_list.split(",") if t.strip()]

    print("=" * 60)
    print("🚀 TradingAgents-Astock Batch Runner (GitHub Actions)")
    print("=" * 60)
    print(f"⏰  Start time : {_now()}")
    print(f"📅  Trade date : {trade_date}")
    print(f"🤖  LLM        : {llm_provider} / {deep_think_llm} / {quick_think_llm}")
    print(f"📊  Tickers    : {len(tickers)} stocks — {', '.join(tickers)}")
    print(f"💬  Language   : {output_language}")
    print(f"🗣️   Debates    : {max_debate_rounds} research / {max_risk_discuss_rounds} risk")
    print()

    # ── Build config ─────────────────────────────────────────────────────
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = deep_think_llm
    config["quick_think_llm"] = quick_think_llm
    config["max_debate_rounds"] = max_debate_rounds
    config["max_risk_discuss_rounds"] = max_risk_discuss_rounds
    config["output_language"] = output_language
    config["results_dir"] = results_base
    if backend_url:
        config["backend_url"] = backend_url
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }

    # ── Run ──────────────────────────────────────────────────────────────
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    results: dict[str, dict] = {}
    passed = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"{'─' * 50}")
        print(f"[{i}/{len(tickers)}] 🔍 Analyzing: {ticker}")
        print(f"{'─' * 50}")

        start = time.time()
        try:
            ta = TradingAgentsGraph(debug=False, config=config)
            final_state, decision = ta.propagate(ticker, trade_date)
            elapsed = time.time() - start

            # Save report
            out_dir = Path(results_base) / ticker / trade_date
            out_dir.mkdir(parents=True, exist_ok=True)

            report_path = out_dir / "decision.md"
            report_path.write_text(
                f"# {ticker} Analysis Report\n\n"
                f"- **Date**: {trade_date}\n"
                f"- **Signal**: {decision}\n"
                f"- **Duration**: {elapsed:.0f}s ({elapsed / 60:.1f} min)\n"
                f"- **LLM**: {llm_provider} / {deep_think_llm} / {quick_think_llm}\n\n"
                f"## Full Decision\n\n"
                f"{final_state.get('final_trade_decision', 'N/A')}",
                encoding="utf-8",
            )

            # JSON summary for programmatic use
            summary_path = out_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "run_time": _now(),
                        "duration_seconds": round(elapsed),
                        "signal": decision,
                        "llm_provider": llm_provider,
                        "decision_preview": final_state.get(
                            "final_trade_decision", ""
                        )[:2000],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"✅ {ticker}: {decision} ({elapsed:.0f}s)")
            results[ticker] = {"status": "ok", "signal": decision, "elapsed": elapsed}
            passed += 1

        except Exception as exc:
            elapsed = time.time() - start
            print(f"❌ {ticker}: FAILED — {exc} ({elapsed:.0f}s)")
            results[ticker] = {"status": "error", "error": str(exc), "elapsed": elapsed}
            failed += 1

    # ── Summary ──────────────────────────────────────────────────────────
    total = len(tickers)
    print()
    print("=" * 60)
    print("📊 Batch Complete")
    print("=" * 60)
    print(f"✅ Passed : {passed} / {total}")
    print(f"❌ Failed : {failed} / {total}")
    print(f"⏰ End    : {_now()}")
    print()

    # Print per-ticker results
    for ticker, info in results.items():
        status_icon = "✅" if info["status"] == "ok" else "❌"
        elapsed_str = f"{info['elapsed']:.0f}s"
        detail = info.get("signal") if info["status"] == "ok" else info.get("error")
        print(f"  {status_icon} {ticker:12s} {elapsed_str:>8s}  {detail}")

    # Write aggregate summary
    summary_dir = Path(results_base)
    summary_dir.mkdir(parents=True, exist_ok=True)
    agg_path = summary_dir / f"batch_summary_{trade_date}.json"
    agg_path.write_text(
        json.dumps(
            {
                "trade_date": trade_date,
                "run_time": _now(),
                "total": total,
                "passed": passed,
                "failed": failed,
                "llm": {
                    "provider": llm_provider,
                    "deep": deep_think_llm,
                    "quick": quick_think_llm,
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n📄 Aggregate summary: {agg_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
