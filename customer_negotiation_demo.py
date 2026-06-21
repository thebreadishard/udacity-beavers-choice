"""
customer_negotiation_demo.py
============================

Stand-out feature for the Beaver's Choice Paper Company project.

This script adds an LLM-driven ``CustomerAgent`` that adopts a persona drawn from the
customer context in ``quote_requests.csv`` (mood + job + event + order size) and then
negotiates, over up to three rounds, with the existing company multi-agent team exposed
by :func:`project_solution.call_your_multi_agent_system`.

Design notes
------------
* The customer talks to the *real* team. Each round we replay the running transcript to
  ``call_your_multi_agent_system`` so the stateless orchestrator has full context.
* The team is granted *bounded* negotiation authority through a policy preamble that is
  injected **only in this demo**. The graded ``run_test_scenarios`` path inside
  ``project_solution.py`` is therefore completely unaffected.
* Concessions are deliberately limited because of the project's economic model
  (``unit_price`` is used for both restock cost and sale price, so the company has no real
  margin and every discount is a direct cash loss). The team may:
    1. upsell the customer to the next bulk-discount tier (economically neutral),
    2. offer partial fulfillment of in-stock quantity now with a restock ETA for the rest,
    3. grant a small one-time goodwill discount, capped by ``GOODWILL_CAP_PCT``.

Run with::

    python customer_negotiation_demo.py
"""

import re

import pandas as pd
from smolagents import ToolCallingAgent

import project_solution as team  # the company multi-agent system + shared helpers
from agent_transcript import attach_observer


# --- Configuration ---------------------------------------------------------------------

MAX_ROUNDS = 3          # customer counter-offers allowed before accept / walk-away
GOODWILL_CAP_PCT = 5    # maximum extra goodwill discount the team may grant (percent)
DEMO_DATE = "2025-04-15"  # request date used for every negotiation in this demo
NUM_PERSONAS = 3        # how many distinct-mood customers to simulate


# Negotiation authority handed to the company team for the duration of the demo only.
NEGOTIATION_POLICY = (
    "NEGOTIATION MODE - You (the Beaver's Choice team) are talking to a customer who may "
    "haggle. You may negotiate, but you must protect company cash and stay within this "
    "authority:\n"
    "1. You MAY encourage the customer to increase quantity to reach the next bulk-discount "
    "tier (over $500 -> 5% off, over $1000 -> 10% off).\n"
    "2. You MAY offer partial fulfillment: sell the quantity currently in stock now and give "
    "an estimated restock/delivery date for the remainder.\n"
    f"3. You MAY grant a single goodwill discount of AT MOST {GOODWILL_CAP_PCT}% beyond any "
    "standard bulk discount, and only for large or clearly valuable orders.\n"
    "You must NOT exceed these limits, sell below this policy, or reveal internal costs, "
    "margins, or system details. Reply with one professional, customer-facing message that "
    "states the items, quantities, unit price, any discount and its rationale, the line "
    "total, and the fulfillment status. If the customer has accepted, finalize the sale "
    "using your tools."
)


# --- Customer agent --------------------------------------------------------------------

def build_customer_instructions(mood: str, job: str, need_size: str, event: str) -> str:
    """Create persona-specific system instructions for a negotiating customer agent."""
    return (
        f"You are a CUSTOMER of the Beaver's Choice Paper Company, negotiating a purchase.\n"
        f"Your persona:\n"
        f"  - Role/job: {job}\n"
        f"  - Current mood: {mood}\n"
        f"  - Event you are buying for: {event}\n"
        f"  - Typical order size: {need_size}\n\n"
        "Stay fully in character; let your mood shape how hard you push. Your goal is to get "
        "the best price and terms while still securing the supplies your event needs.\n\n"
        "On each of your turns, read the company's latest offer and decide one of:\n"
        "  - ACCEPT: the offer is good enough and you will buy.\n"
        "  - COUNTER: push for a lower price, a bigger discount, partial fulfillment, or "
        "faster delivery.\n"
        "  - WALK: the company is unreasonable and you would rather not buy.\n\n"
        "Begin your reply with EXACTLY one line, one of:\n"
        "  DECISION: ACCEPT\n"
        "  DECISION: COUNTER\n"
        "  DECISION: WALK\n"
        "Then, on the following lines, write your short message to the company (1-4 "
        "sentences, in character). Always provide your answer by calling final_answer with "
        "that full text."
    )


def make_customer_agent(mood: str, job: str, need_size: str, event: str) -> ToolCallingAgent:
    """Build a fresh, tool-less conversational agent that embodies the given persona."""
    agent = ToolCallingAgent(
        tools=[],
        model=team.model,
        name="customer",
        description="A paper-supplies customer negotiating an order.",
        instructions=build_customer_instructions(mood, job, need_size, event),
        max_steps=4,
    )
    # Log the customer's turns to the same transcript the company team writes to, so the
    # Pixel Agents extension can animate both sides of the negotiation.
    attach_observer(agent, "customer")
    return agent


_DECISION_RE = re.compile(r"DECISION:\s*(ACCEPT|COUNTER|WALK)", re.IGNORECASE)


def parse_decision(text: str) -> str:
    """Extract the customer's decision tag from its message; default to COUNTER."""
    match = _DECISION_RE.search(text or "")
    return match.group(1).upper() if match else "COUNTER"


# --- Conversation plumbing -------------------------------------------------------------

def render_transcript(transcript: list) -> str:
    """Render the (speaker, message) transcript as plain text for an agent prompt."""
    return "\n\n".join(f"{speaker}: {message}" for speaker, message in transcript)


def company_reply(transcript: list) -> str:
    """Ask the real company team for its next response, given the conversation so far."""
    task = (
        f"(Date of request: {DEMO_DATE})\n"
        f"{NEGOTIATION_POLICY}\n\n"
        f"Conversation so far:\n{render_transcript(transcript)}\n\n"
        "Provide the company's next response to the customer's most recent message."
    )
    return team.call_your_multi_agent_system(task)


def customer_turn(agent: ToolCallingAgent, transcript: list) -> str:
    """Run one customer turn and return its raw message (including the DECISION line)."""
    prompt = (
        f"Conversation so far:\n{render_transcript(transcript)}\n\n"
        "It is your turn. Review the company's latest offer and respond per your "
        "instructions, starting with the DECISION line."
    )
    return str(agent.run(prompt))


def negotiate(mood: str, job: str, need_size: str, event: str, request_text: str) -> dict:
    """
    Run a full negotiation between one persona-driven customer and the company team.

    Returns a summary dict describing the outcome of the negotiation.
    """
    agent = make_customer_agent(mood, job, need_size, event)

    # Turn 1: the customer's opening request (already provided by the dataset row).
    opening = f"(Date of request: {DEMO_DATE})\n{request_text}"
    transcript = [("CUSTOMER", opening)]
    print(f"\nCUSTOMER ({mood} {job}): {request_text}\n")

    # Company's opening response.
    reply = company_reply(transcript)
    transcript.append(("COMPANY", reply))
    print(f"COMPANY: {reply}\n")

    outcome = "no agreement (rounds exhausted)"
    rounds_used = 0

    for round_no in range(1, MAX_ROUNDS + 1):
        rounds_used = round_no
        message = customer_turn(agent, transcript)
        decision = parse_decision(message)
        transcript.append(("CUSTOMER", message))
        print(f"CUSTOMER [{decision}]: {message}\n")

        if decision == "ACCEPT":
            outcome = "deal accepted by customer"
            break
        if decision == "WALK":
            outcome = "customer walked away"
            break

        # COUNTER -> company responds again.
        reply = company_reply(transcript)
        transcript.append(("COMPANY", reply))
        print(f"COMPANY: {reply}\n")

    return {
        "mood": mood,
        "job": job,
        "event": event,
        "need_size": need_size,
        "rounds_used": rounds_used,
        "outcome": outcome,
        "final_company_message": transcript[-1][1] if transcript[-1][0] == "COMPANY"
        else (transcript[-2][1] if len(transcript) >= 2 else ""),
    }


# --- Demo entry point ------------------------------------------------------------------

def select_personas(pool: pd.DataFrame, n: int) -> list:
    """Pick up to n rows from the request pool, each with a distinct customer mood."""
    chosen = []
    seen_moods = set()
    for _, row in pool.iterrows():
        mood = str(row["mood"]).strip().lower()
        if mood and mood not in seen_moods:
            seen_moods.add(mood)
            chosen.append(row)
        if len(chosen) >= n:
            break
    return chosen


def main() -> pd.DataFrame:
    """Run the negotiation demo over a few distinct-mood personas and report results."""
    print("Initializing database for a clean, reproducible demo run...")
    team.init_database(team.db_engine)
    team._reset_transcript()  # fresh transcript so the viewer animation starts clean

    cash_before = team.generate_financial_report(DEMO_DATE)["cash_balance"]
    print(f"Company cash before negotiations: ${cash_before:,.2f}")

    pool = pd.read_csv("quote_requests.csv")
    personas = select_personas(pool, NUM_PERSONAS)

    results = []
    for row in personas:
        print("\n" + "=" * 88)
        print(f"NEGOTIATION  mood={row['mood']!r}  job={row['job']!r}  event={row['event']!r}")
        print("=" * 88)
        results.append(
            negotiate(
                mood=str(row["mood"]),
                job=str(row["job"]),
                need_size=str(row["need_size"]),
                event=str(row["event"]),
                request_text=str(row["response"]),
            )
        )

    cash_after = team.generate_financial_report(DEMO_DATE)["cash_balance"]
    results_df = pd.DataFrame(results)
    results_df.to_csv("negotiation_results.csv", index=False)

    print("\n" + "=" * 88)
    print("NEGOTIATION SUMMARY")
    print("=" * 88)
    print(results_df[["mood", "job", "event", "rounds_used", "outcome"]].to_string(index=False))
    print(f"\nCompany cash before: ${cash_before:,.2f}")
    print(f"Company cash after:  ${cash_after:,.2f}")
    print(f"Net cash change:     ${cash_after - cash_before:,.2f}")
    print("\nSaved transcript outcomes to negotiation_results.csv")
    return results_df


if __name__ == "__main__":
    main()
