"""LangGraph bilateral state machine orchestrating A2A vendor negotiations."""

from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, END
from schemas import NegotiationState, Offer
from agents import buyer_agent_node, supplier_agent_node


def check_buyer_transition(state: NegotiationState) -> Literal["supplier", "__end__"]:
    """Evaluate outcome after Buyer turn."""
    if state.get("status") == "accepted":
        return END
    if state.get("status") == "rejected":
        return END
    if state["turn_count"] >= state.get("turn_limit", 6):
        return END
    return "supplier"


def check_supplier_transition(state: NegotiationState) -> Literal["buyer", "__end__"]:
    """Evaluate outcome after Supplier turn."""
    if state.get("status") == "accepted":
        return END
    if state.get("status") == "rejected":
        return END
    if state["turn_count"] >= state.get("turn_limit", 6):
        return END
    return "buyer"


def build_negotiation_graph():
    """Construct and compile the LangGraph A2A negotiation workflow."""
    workflow = StateGraph(NegotiationState)

    # Register nodes
    workflow.add_node("buyer", buyer_agent_node)
    workflow.add_node("supplier", supplier_agent_node)

    # Set entry point: Supplier evaluates initial buyer proposal, or Buyer begins
    workflow.set_entry_point("supplier")

    # Add conditional alternating edges
    workflow.add_conditional_edges(
        "supplier",
        check_supplier_transition,
        {
            "buyer": "buyer",
            END: END,
        }
    )

    workflow.add_conditional_edges(
        "buyer",
        check_buyer_transition,
        {
            "supplier": "supplier",
            END: END,
        }
    )

    return workflow.compile()


# Pre-compiled graph instance
negotiation_graph = build_negotiation_graph()


def run_negotiation_simulation(
    product_name: str = "Aashirvaad Superior MP Whole Wheat Atta 5kg",
    initial_offer: Offer = Offer(price_per_unit=3.90, volume=8000, delivery_days=3),
    turn_limit: int = 6,
) -> Dict[str, Any]:
    """Execute complete A2A negotiation simulation from starting parameters."""
    starting_message = {
        "turn": 1,
        "speaker": "Buyer (Milkbasket)",
        "action": "propose_counter",
        "offer": initial_offer.model_dump(),
        "message": (
            f"Milkbasket opening proposal for {product_name}: "
            f"${initial_offer.price_per_unit:.2f}/unit for {initial_offer.volume:,} units "
            f"with {initial_offer.delivery_days}-day delivery lead time."
        ),
    }

    initial_state: NegotiationState = {
        "messages": [starting_message],
        "current_offer": initial_offer.model_dump(),
        "last_offer": None,
        "status": "negotiating",
        "turn_count": 1,
        "turn_limit": turn_limit,
        "product_name": product_name,
        "agreement_summary": None,
    }

    final_state = negotiation_graph.invoke(initial_state)

    # Handle turn-limit expiry without explicit acceptance
    if final_state["status"] == "negotiating" and final_state["turn_count"] >= turn_limit:
        final_state["status"] = "rejected"
        final_state["agreement_summary"] = (
            f"Negotiation ended in deadlock: Maximum turn limit ({turn_limit} turns) reached "
            f"without mutual contract agreement."
        )

    return final_state
