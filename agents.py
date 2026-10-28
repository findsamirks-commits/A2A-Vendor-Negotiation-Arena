"""Autonomous Negotiating Agents: Retail Buyer (Milkbasket) and FMCG Supplier.
Powered by LangChain and Groq's OpenAI-compatible endpoint with openai/gpt-oss-120b.
Both agents reason over bilateral commercial incentives and output structured Pydantic terms.
"""

import os
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schemas import AgentTurnResponse, NegotiationState, Offer

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

# --- Agent Prompts ---
BUYER_SYSTEM_PROMPT = """You are the Senior Category Sourcing Lead for Milkbasket, an enterprise quick-commerce grocery delivery platform.
Your mandate is to procure inventory from FMCG suppliers under strict commercial guardrails.

Commercial Rules & Constraints for Buyer:
1. Price Ceiling: You are STRICTLY FORBIDDEN from agreeing to any price of $4.50 or higher.
   - Ideal target price: $3.80 – $4.10 per unit.
2. Volume Envelope: You must strictly keep total order volume UNDER 10,000 units (max 9,500 units).
   - Your warehouse cannot take more than 10,000 units due to working capital and shelf-life constraints.
3. Delivery SLA: You demand delivery within 3 to 5 business days to avoid dark store stockouts.
4. Acceptance Decision (action='accept'):
   - If the supplier's latest offer satisfies: Price <= $4.20 AND Volume <= 10,000 AND Delivery <= 5 days, you should ACCEPT the deal.
   - If the supplier's terms are acceptable, set action='accept' and match the agreed price, volume, and delivery_days.
5. Counter-Offer Decision (action='propose_counter'):
   - If price > $4.20 or volume > 10,000, counter with a competitive offer (e.g. increase volume slightly towards 7,000-8,500 in exchange for price drop to $4.05-$4.18).
6. Walk-Away (action='reject'):
   - Only reject if supplier insists on price >= $4.50 or volume > 10,000 with zero concessions.

Tone: Professional, commercially savvy, assertive retail buyer protecting Milkbasket's gross margin.
"""

SUPPLIER_SYSTEM_PROMPT = """You are the National Key Accounts Director for a tier-1 FMCG food and dairy manufacturer negotiating with Milkbasket.
Your mandate is to secure profitable volume commitments while protecting manufacturing margins.

Commercial Rules & Constraints for Supplier:
1. Price Floor: You are STRICTLY FORBIDDEN from agreeing to any price below $4.00.
   - Selling below $4.00 results in direct gross margin loss for your manufacturing plant.
   - Target price: $4.25 – $4.40 per unit.
2. Minimum Order Quantity (MOQ): You require a STRICT MINIMUM of 5,000 units.
   - You push for higher volumes (6,000 to 9,000 units) to justify bulk price discounts.
3. Delivery SLA: Production and logistics lead time requires 4 to 7 business days.
4. Acceptance Decision (action='accept'):
   - If the buyer's latest offer satisfies: Price >= $4.15 AND Volume >= 6,000 AND Delivery >= 4 days, you should ACCEPT the deal.
   - If the buyer's terms are acceptable, set action='accept' and match the agreed price, volume, and delivery_days.
5. Counter-Offer Decision (action='propose_counter'):
   - If price < $4.15 or volume < 6,000, propose a counter-offer offering volume discounts (e.g., offer $4.18 if buyer commits to 8,000 units, or $4.22 for 6,500 units).
6. Walk-Away (action='reject'):
   - Only reject if buyer refuses to pay >= $4.00 or refuses the 5,000 unit MOQ.

Tone: Professional, value-driven manufacturer negotiating bulk contractual pricing.
"""


def get_groq_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    base_url: str = DEFAULT_BASE_URL,
) -> ChatOpenAI:
    """Initialize ChatOpenAI configured for Groq's endpoint with openai/gpt-oss-120b."""
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key or "placeholder-groq-key",
        base_url=base_url,
    )


def format_transcript_context(messages: list) -> str:
    """Format dialogue history into readable conversational context for LLM prompt."""
    if not messages:
        return "No previous exchanges. You are opening the negotiation."
    lines = []
    for m in messages:
        lines.append(
            f"Turn {m['turn']} [{m['speaker']}]: Action='{m['action']}' | "
            f"Terms: ${m['offer']['price_per_unit']:.2f}/unit, {m['offer']['volume']:,} units, "
            f"{m['offer']['delivery_days']} days | \"{m['message']}\""
        )
    return "\n".join(lines)


# --- Deterministic Fallback Logic ---
def fallback_buyer_turn(state: NegotiationState) -> AgentTurnResponse:
    """Algorithmic concession ladder for Retail Buyer Agent."""
    last_offer = state.get("current_offer") or {}
    p_last = last_offer.get("price_per_unit", 4.35)
    v_last = last_offer.get("volume", 7000)
    d_last = last_offer.get("delivery_days", 5)

    # Acceptance check: Price <= $4.20 and Vol <= 10,000
    if p_last <= 4.20 and v_last <= 10000:
        return AgentTurnResponse(
            action="accept",
            price_per_unit=round(p_last, 2),
            volume=v_last,
            delivery_days=min(d_last, 5),
            conversational_rationale=(
                f"We accept your proposed price of ${p_last:.2f} for {v_last:,} units with "
                f"{d_last}-day delivery. This fits within Milkbasket's margin parameters."
            ),
        )

    # Concession counter: step price up towards $4.15 - $4.18, increase volume towards 8,000
    turn = state["turn_count"]
    target_price = min(4.18, 3.90 + (turn * 0.08))
    target_vol = min(9000, 7000 + (turn * 400))
    target_days = 4

    return AgentTurnResponse(
        action="propose_counter",
        price_per_unit=round(target_price, 2),
        volume=target_vol,
        delivery_days=target_days,
        conversational_rationale=(
            f"Your request of ${p_last:.2f} exceeds our retail margin envelope. We can increase our order "
            f"commitment to {target_vol:,} units if you meet us at ${target_price:.2f} with a {target_days}-day lead time."
        ),
    )


def fallback_supplier_turn(state: NegotiationState) -> AgentTurnResponse:
    """Algorithmic concession ladder for FMCG Supplier Agent."""
    last_offer = state.get("current_offer") or {}
    p_last = last_offer.get("price_per_unit", 3.90)
    v_last = last_offer.get("volume", 7000)
    d_last = last_offer.get("delivery_days", 4)

    # Acceptance check: Price >= $4.15 and Vol >= 6,000
    if p_last >= 4.15 and v_last >= 6000:
        return AgentTurnResponse(
            action="accept",
            price_per_unit=round(p_last, 2),
            volume=v_last,
            delivery_days=max(d_last, 4),
            conversational_rationale=(
                f"Deal confirmed. We accept ${p_last:.2f} per unit for {v_last:,} units with "
                f"{d_last}-day delivery. We will allocate production capacity immediately."
            ),
        )

    # Concession counter: step price down towards $4.18, require volume >= 7,500
    turn = state["turn_count"]
    target_price = max(4.15, 4.38 - (turn * 0.06))
    target_vol = max(6500, min(9500, v_last + 500))
    target_days = 5

    return AgentTurnResponse(
        action="propose_counter",
        price_per_unit=round(target_price, 2),
        volume=target_vol,
        delivery_days=target_days,
        conversational_rationale=(
            f"A price of ${p_last:.2f} is below our manufacturing breakeven. However, for a committed volume "
            f"of {target_vol:,} units, we can discount our price to ${target_price:.2f} with {target_days} days delivery."
        ),
    )


# --- Node Execution Functions ---
def buyer_agent_node(state: NegotiationState) -> Dict[str, Any]:
    """Execute Retail Buyer Agent turn in LangGraph."""
    state["turn_count"] += 1
    current_turn = state["turn_count"]
    product = state["product_name"]
    transcript = format_transcript_context(state["messages"])

    try:
        llm = get_groq_llm()
        structured_llm = llm.with_structured_output(AgentTurnResponse)
        prompt = ChatPromptTemplate.from_messages([
            ("system", BUYER_SYSTEM_PROMPT),
            ("user", f"""Product under negotiation: {product}
Turn Number: {current_turn} of {state['turn_limit']}
Current Dialogue Transcript:
{transcript}

Evaluate the supplier's position and output your structured AgentTurnResponse.""")
        ])
        chain = prompt | structured_llm
        turn_response: AgentTurnResponse = chain.invoke({})
    except Exception:
        turn_response = fallback_buyer_turn(state)

    offer_dict = {
        "price_per_unit": round(turn_response.price_per_unit, 2),
        "volume": turn_response.volume,
        "delivery_days": turn_response.delivery_days,
    }

    message_entry = {
        "turn": current_turn,
        "speaker": "Buyer (Milkbasket)",
        "action": turn_response.action,
        "offer": offer_dict,
        "message": turn_response.conversational_rationale,
    }

    # Status determination
    if turn_response.action == "accept":
        new_status = "accepted"
        summary = (
            f"Agreement achieved on Turn {current_turn}: Buyer accepted supplier terms at "
            f"${offer_dict['price_per_unit']:.2f}/unit for {offer_dict['volume']:,} units."
        )
    elif turn_response.action == "reject":
        new_status = "rejected"
        summary = f"Negotiation terminated by Buyer on Turn {current_turn} due to unacceptable terms."
    else:
        new_status = "negotiating"
        summary = None

    return {
        "messages": state["messages"] + [message_entry],
        "last_offer": state.get("current_offer"),
        "current_offer": offer_dict,
        "status": new_status,
        "turn_count": current_turn,
        "agreement_summary": summary or state.get("agreement_summary"),
    }


def supplier_agent_node(state: NegotiationState) -> Dict[str, Any]:
    """Execute FMCG Supplier Agent turn in LangGraph."""
    state["turn_count"] += 1
    current_turn = state["turn_count"]
    product = state["product_name"]
    transcript = format_transcript_context(state["messages"])

    try:
        llm = get_groq_llm()
        structured_llm = llm.with_structured_output(AgentTurnResponse)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPPLIER_SYSTEM_PROMPT),
            ("user", f"""Product under negotiation: {product}
Turn Number: {current_turn} of {state['turn_limit']}
Current Dialogue Transcript:
{transcript}

Evaluate the buyer's proposal and output your structured AgentTurnResponse.""")
        ])
        chain = prompt | structured_llm
        turn_response: AgentTurnResponse = chain.invoke({})
    except Exception:
        turn_response = fallback_supplier_turn(state)

    offer_dict = {
        "price_per_unit": round(turn_response.price_per_unit, 2),
        "volume": turn_response.volume,
        "delivery_days": turn_response.delivery_days,
    }

    message_entry = {
        "turn": current_turn,
        "speaker": "Supplier (FMCG Vendor)",
        "action": turn_response.action,
        "offer": offer_dict,
        "message": turn_response.conversational_rationale,
    }

    # Status determination
    if turn_response.action == "accept":
        new_status = "accepted"
        summary = (
            f"Agreement achieved on Turn {current_turn}: Supplier accepted buyer terms at "
            f"${offer_dict['price_per_unit']:.2f}/unit for {offer_dict['volume']:,} units."
        )
    elif turn_response.action == "reject":
        new_status = "rejected"
        summary = f"Negotiation terminated by Supplier on Turn {current_turn} due to price/volume impasse."
    else:
        new_status = "negotiating"
        summary = None

    return {
        "messages": state["messages"] + [message_entry],
        "last_offer": state.get("current_offer"),
        "current_offer": offer_dict,
        "status": new_status,
        "turn_count": current_turn,
        "agreement_summary": summary or state.get("agreement_summary"),
    }
