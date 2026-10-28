"""Pydantic schemas and LangGraph state definitions for A2A Vendor Negotiation."""

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class Offer(BaseModel):
    """Structured commercial terms exchanged between negotiating agents."""

    price_per_unit: float = Field(
        ge=0.01,
        description="Agreed or proposed purchase price per unit in USD"
    )
    volume: int = Field(
        ge=1,
        description="Total order quantity in units"
    )
    delivery_days: int = Field(
        ge=1,
        le=60,
        description="Fulfillment lead time in business days from Purchase Order confirmation"
    )


class AgentTurnResponse(BaseModel):
    """Structured decision output produced by an individual negotiating agent."""

    action: Literal["propose_counter", "accept", "reject"] = Field(
        description="Negotiation decision: 'accept' if the counterparty's offer is satisfactory, 'propose_counter' to submit revised terms, or 'reject' to terminate negotiations."
    )
    price_per_unit: float = Field(
        description="The unit price offered or agreed upon in USD"
    )
    volume: int = Field(
        description="The order quantity offered or agreed upon in units"
    )
    delivery_days: int = Field(
        description="The fulfillment delivery window in days"
    )
    conversational_rationale: str = Field(
        description="Natural language commercial reasoning sent to the counterparty explaining the position, concessions, or acceptance."
    )


class NegotiationMessage(BaseModel):
    """Transcript entry recorded during the negotiation exchange."""

    turn: int = Field(description="Sequential turn index (1 to 6)")
    speaker: Literal["Buyer (Milkbasket)", "Supplier (FMCG Vendor)"] = Field(
        description="The agent who spoke this turn"
    )
    action: Literal["propose_counter", "accept", "reject"] = Field(
        description="Action taken by the speaker"
    )
    offer: Offer = Field(description="Commercial terms offered or accepted")
    message: str = Field(description="Conversational statement from the agent")


class NegotiationState(TypedDict):
    """LangGraph state schema maintaining the bilateral negotiation context."""

    messages: List[Dict[str, Any]]
    current_offer: Optional[Dict[str, Any]]
    last_offer: Optional[Dict[str, Any]]
    status: Literal["negotiating", "accepted", "rejected"]
    turn_count: int
    turn_limit: int
    product_name: str
    agreement_summary: Optional[str]


class NegotiationRequest(BaseModel):
    """FastAPI request body for initiating an A2A negotiation session."""

    product_name: str = Field(
        default="Aashirvaad Superior MP Whole Wheat Atta 5kg",
        description="The retail SKU undergoing vendor procurement negotiation"
    )
    initial_offer: Offer = Field(
        default=Offer(price_per_unit=3.90, volume=8000, delivery_days=3),
        description="Starting proposal initiated by the Retail Buyer"
    )
    turn_limit: int = Field(
        default=6,
        ge=2,
        le=12,
        description="Maximum turns before declaring deadlock / status='rejected'"
    )


class NegotiationResponse(BaseModel):
    """FastAPI response payload returning final contract status and full audit transcript."""

    status: Literal["negotiating", "accepted", "rejected"] = Field(
        description="Final negotiation outcome ('accepted' or 'rejected')"
    )
    product_name: str = Field(description="The negotiated product SKU")
    total_turns: int = Field(description="Total number of conversational turns completed")
    final_agreed_terms: Optional[Offer] = Field(
        description="Final contract terms if accepted, None if rejected"
    )
    agreement_summary: str = Field(
        description="Executive summary of the commercial resolution or failure mode"
    )
    transcript: List[NegotiationMessage] = Field(
        description="Full turn-by-turn dialogue transcript between the two autonomous agents"
    )
