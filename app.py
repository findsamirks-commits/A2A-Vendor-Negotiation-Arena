"""Streamlit Frontend for Autonomous A2A Vendor Negotiation Arena.

Interactive UI for simulating and visualizing bilateral commercial negotiations
between Milkbasket's Retail Buyer Agent and FMCG Supplier Agent.
"""

import streamlit as st
import requests

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="A2A Vendor Negotiation Arena",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge-buyer {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-supplier {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🤝 Autonomous A2A Vendor Negotiation Arena</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "Autonomous bilateral procurement bargaining between <b>Milkbasket Buyer Agent</b> "
    "and <b>FMCG Supplier Agent</b> via LangGraph & Groq (<code>openai/gpt-oss-120b</code>)."
    "</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Commercial Parameters & Initial Offer
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Negotiation Controls")
    st.caption("Configure initial terms proposed by Milkbasket Category Sourcing.")

    product_name = st.text_input(
        "Product SKU",
        value="Aashirvaad Superior MP Whole Wheat Atta 5kg",
        help="Target retail item undergoing vendor contract bargaining",
    )

    st.subheader("Buyer Starting Offer")

    col_p, col_v = st.columns(2)
    with col_p:
        price_per_unit = st.number_input(
            "Price / Unit ($)",
            min_value=0.50,
            max_value=10.00,
            value=3.90,
            step=0.05,
            format="%.2f",
            help="Starting unit price offered to vendor (Ceiling: $4.50)",
        )
    with col_v:
        volume = st.number_input(
            "Order Volume",
            min_value=500,
            max_value=15000,
            value=8000,
            step=500,
            help="Units to commit in Purchase Order (Cap: 10,000)",
        )

    delivery_days = st.slider(
        "Delivery SLA (Days)",
        min_value=1,
        max_value=14,
        value=3,
        help="Target fulfillment window in days",
    )

    st.markdown("---")
    st.subheader("⚙️ Engine Settings")
    api_url = st.text_input(
        "FastAPI Endpoint",
        value="http://127.0.0.1:8000/negotiate",
        help="Backend LangGraph FastAPI server URL",
    )

    turn_limit = st.slider(
        "Turn Limit",
        min_value=2,
        max_value=10,
        value=6,
        help="Maximum rounds before deadlock is triggered",
    )

    st.markdown("---")
    start_btn = st.button("🚀 Start Negotiation", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📋 Commercial Guardrails")
    st.markdown(
        """
        - **Buyer (Milkbasket):**
          - Hard Ceiling: **$4.50**
          - Max Volume: **10,000 units**
        - **Supplier (FMCG Vendor):**
          - Hard Floor: **$4.00**
          - Minimum MOQ: **5,000 units**
        - **ZOPA Range:** $4.15 – $4.20
        """
    )

# -----------------------------------------------------------------------------
# Main Content & Negotiation Flow
# -----------------------------------------------------------------------------
if start_btn:
    payload = {
        "product_name": product_name,
        "initial_offer": {
            "price_per_unit": float(price_per_unit),
            "volume": int(volume),
            "delivery_days": int(delivery_days),
        },
        "turn_limit": int(turn_limit),
    }

    st.subheader(f"📦 Active Negotiation: {product_name}")
    st.caption(
        f"Initial Opening Bid: **${price_per_unit:.2f}/unit** | "
        f"**{volume:,} units** | **{delivery_days} days delivery**"
    )

    with st.spinner("🤖 Agents negotiating terms via LangGraph state machine..."):
        try:
            response = requests.post(api_url, json=payload, timeout=60)
            if response.status_code != 200:
                st.error(f"❌ Backend error ({response.status_code}): {response.text}")
                st.stop()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ **Could not connect to FastAPI server at `http://127.0.0.1:8000`**.\n\n"
                "Please start the backend server in your terminal:\n"
                "```bash\nuvicorn main:app --host 127.0.0.1 --port 8000 --reload\n```"
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

    # -------------------------------------------------------------------------
    # Parse & Display Transcript with st.chat_message
    # -------------------------------------------------------------------------
    transcript = data.get("transcript", [])
    status = data.get("status", "negotiating")
    total_turns = data.get("total_turns", len(transcript))
    final_terms = data.get("final_agreed_terms")
    agreement_summary = data.get("agreement_summary", "")

    st.markdown("### 💬 Bilateral Agent Transcript")

    for msg in transcript:
        turn = msg.get("turn", 1)
        speaker = msg.get("speaker", "Agent")
        action = msg.get("action", "propose_counter").upper()
        offer = msg.get("offer", {})
        message_text = msg.get("message", "")

        p = offer.get("price_per_unit", 0.0)
        v = offer.get("volume", 0)
        d = offer.get("delivery_days", 0)
        po_value = p * v

        if "Buyer" in speaker:
            with st.chat_message("user", avatar="🛒"):
                st.markdown(
                    f"**Turn {turn}** • <span class='badge-buyer'>🛒 {speaker}</span> "
                    f"• Action: **`{action}`**",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> *\"{message_text}\"*")
                st.markdown(
                    f"📊 **Proposed Terms:** `${p:.2f}/unit` | `{v:,} units` | `{d} days delivery` "
                    f"*(PO Value: ${po_value:,.2f})*"
                )
        else:
            with st.chat_message("assistant", avatar="🏭"):
                st.markdown(
                    f"**Turn {turn}** • <span class='badge-supplier'>🏭 {speaker}</span> "
                    f"• Action: **`{action}`**",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> *\"{message_text}\"*")
                st.markdown(
                    f"📊 **Proposed Terms:** `${p:.2f}/unit` | `{v:,} units` | `{d} days delivery` "
                    f"*(PO Value: ${po_value:,.2f})*"
                )

    # -------------------------------------------------------------------------
    # Final Result & Metric Box
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏁 Final Resolution")

    if status == "accepted" and final_terms:
        fin_p = final_terms.get("price_per_unit", 0.0)
        fin_v = final_terms.get("volume", 0)
        fin_d = final_terms.get("delivery_days", 0)
        total_contract_value = fin_p * fin_v

        st.success(f"### ✅ AGREEMENT REACHED IN {total_turns} ROUNDS\n\n{agreement_summary}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Price / Unit", f"${fin_p:.2f}")
        m2.metric("Agreed Volume", f"{fin_v:,} units")
        m3.metric("Delivery SLA", f"{fin_d} days")
        m4.metric("Total Contract Value", f"${total_contract_value:,.2f}")

    else:
        st.error(f"### ❌ NEGOTIATION DEADLOCK / REJECTED\n\n{agreement_summary}")
        m1, m2 = st.columns(2)
        m1.metric("Status", "REJECTED")
        m2.metric("Rounds Exceeded", f"{total_turns} / {turn_limit}")

else:
    # Landing instructions
    st.info(
        "👈 **Configure the initial offer in the sidebar and click 'Start Negotiation'** "
        "to trigger the autonomous bilateral bargaining arena."
    )
    with st.expander("ℹ️ How the A2A State Machine Works", expanded=True):
        st.markdown(
            """
            1. **Retail Buyer Agent (Milkbasket):** Strives for the lowest unit price, capping order volume at 10,000 units and strictly rejecting any price $\\ge \\$4.50$.
            2. **FMCG Supplier Agent (Vendor):** Defends commercial margins by refusing any price $< \\$4.00$, enforces a minimum batch commitment of 5,000 units, and trades volume scale for price concessions.
            3. **LangGraph State Machine:** Continues counter-offering until either party accepts the previous terms or the 6-turn limit is reached (triggering deadlock rejection).
            """
        )
