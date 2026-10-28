"""Autonomous A2A Vendor Negotiation Arena - FastAPI Application & CLI Runner.
Exposes POST /negotiate endpoint and provides a standalone simulation CLI with formatted tabular reporting.
"""

import os
import sys
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tabulate import tabulate

# Cross-platform UTF-8 terminal handling
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from schemas import (
    NegotiationRequest,
    NegotiationResponse,
    NegotiationMessage,
    Offer,
)
from graph import run_negotiation_simulation

app = FastAPI(
    title="Autonomous A2A Vendor Negotiation Arena",
    description="Multi-agent bilateral procurement state machine powered by LangGraph and Groq (openai/gpt-oss-120b)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Service status and API documentation entry point."""
    return {
        "service": "Autonomous A2A Vendor Negotiation Arena",
        "status": "online",
        "model": "openai/gpt-oss-120b (Groq)",
        "endpoints": {
            "POST /negotiate": "Initiate bilateral agent-to-agent procurement negotiation",
            "GET /docs": "Interactive Swagger API documentation",
        },
    }


@app.get("/health")
def health_check():
    """Health check probe."""
    return {"status": "healthy"}


@app.post("/negotiate", response_model=NegotiationResponse)
def negotiate(request: NegotiationRequest):
    """Execute LangGraph bilateral negotiation between Retail Buyer and FMCG Supplier."""
    try:
        final_state = run_negotiation_simulation(
            product_name=request.product_name,
            initial_offer=request.initial_offer,
            turn_limit=request.turn_limit,
        )

        transcript_objects = [
            NegotiationMessage(
                turn=m["turn"],
                speaker=m["speaker"],
                action=m["action"],
                offer=Offer(**m["offer"]),
                message=m["message"],
            )
            for m in final_state["messages"]
        ]

        final_terms = Offer(**final_state["current_offer"]) if final_state["status"] == "accepted" else None

        return NegotiationResponse(
            status=final_state["status"],
            product_name=final_state["product_name"],
            total_turns=final_state["turn_count"],
            final_agreed_terms=final_terms,
            agreement_summary=final_state.get("agreement_summary") or "Negotiation concluded.",
            transcript=transcript_objects,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Negotiation pipeline error: {str(e)}")


def print_console_transcript(final_state: Dict[str, Any]):
    """Format and print an executive turn-by-turn dialogue scorecard to the console."""
    print("\n" + "=" * 125)
    print("                AUTONOMOUS A2A VENDOR NEGOTIATION ARENA - BILATERAL DISPATCH SCORECARD                ")
    print("                Retail Buyer (Milkbasket) <---> FMCG Manufacturer (Tier-1 Supplier)                   ")
    print("=" * 125)

    rows = []
    for m in final_state["messages"]:
        off = m["offer"]
        terms_str = f"${off['price_per_unit']:.2f} / unit\n{off['volume']:,} units\n{off['delivery_days']} days"
        short_msg = (m["message"][:70] + "..") if len(m["message"]) > 72 else m["message"]

        rows.append([
            f"Turn {m['turn']}",
            m["speaker"],
            m["action"].upper(),
            terms_str,
            short_msg,
        ])

    headers = ["Turn", "Agent", "Action", "Proposed Terms", "Conversational Rationale"]
    print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left", numalign="center"))

    status = final_state["status"].upper()
    print("\n" + "=" * 65)
    print(f"            FINAL NEGOTIATION OUTCOME: [{status}]            ")
    print("=" * 65)

    summary_table = [
        ["Product Negotiated", final_state["product_name"]],
        ["Negotiation Status", status],
        ["Total Conversational Turns", f"{final_state['turn_count']} turns"],
        ["Resolution Summary", final_state.get("agreement_summary", "Completed")],
    ]

    if final_state["status"] == "accepted" and final_state.get("current_offer"):
        off = final_state["current_offer"]
        total_p = off["price_per_unit"] * off["volume"]
        summary_table.extend([
            ["Agreed Unit Price", f"${off['price_per_unit']:.2f}"],
            ["Contract Order Volume", f"{off['volume']:,} units"],
            ["Total Purchase Order Value", f"${total_p:,.2f}"],
            ["Committed Lead Time", f"{off['delivery_days']} business days"],
        ])

    print(tabulate(summary_table, headers=["Negotiation Parameter", "Contract Detail"], tablefmt="grid"))
    print("=" * 65 + "\n")


def run_standalone_test():
    """CLI test execution when running main.py directly."""
    print("\n[Init] Initializing Autonomous A2A Negotiation test execution...")
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if groq_key:
        print("[Status] Connected to Groq API endpoint: https://api.groq.com/openai/v1 (openai/gpt-oss-120b)")
    else:
        print("[Notice] No GROQ_API_KEY detected in environment. Running with deterministic bilateral game logic.")

    sample_product = "Aashirvaad Superior MP Whole Wheat Atta 5kg"
    sample_offer = Offer(price_per_unit=3.90, volume=8000, delivery_days=3)

    final_state = run_negotiation_simulation(
        product_name=sample_product,
        initial_offer=sample_offer,
        turn_limit=6,
    )

    print_console_transcript(final_state)


if __name__ == "__main__":
    if "--server" in sys.argv:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_standalone_test()
