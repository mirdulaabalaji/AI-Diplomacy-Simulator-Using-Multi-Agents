"""
Two-phase agent logic:
  Phase 1 — MESSAGING: Each agent composes a private diplomatic message
      to every other agent. Each message is recipient-specific and
      capped at 280 characters to force concise, strategic communication.
  Phase 2 — DECISION: Each agent reads its inbox and chooses C or D.
"""

from schemas.state import NegotiationState, CountryMemory, Message
from agents.llm import invoke_llm
from memory.long_term_memory import SimulationMemory

# Character limit per diplomatic message — forces agents to be concise
# and prevents the LLM from producing generic waffle
MESSAGE_CHAR_LIMIT = 280
_lt_memory = SimulationMemory()

# ---------------------------------------------------------------------------
# PHASE 1: MESSAGING
# ---------------------------------------------------------------------------

def compose_message(
    state: NegotiationState,
    sender_id: str,
    recipient_id: str,
) -> Message:
    sender = state.countries[sender_id]
    recipient = state.countries[recipient_id]

    memory = sender.memory.get(recipient_id)
    memory_summary = (
        f"Cooperation rate: {memory.cooperation_rate:.2f}, "
        f"Betrayals: {memory.betrayal_count}, "
        f"Alliances: {memory.alliance_count}"
        if memory else "No prior history."
    )

    lt_context = _lt_memory.get_agent_history(recipient.name)
    lt_summary = (
        f"Appeared in {lt_context['appearances']} past simulation(s). "
        f"Average payoff: {lt_context['avg_payoff']}, "
        f"cooperation rate: {int(lt_context['avg_coop_rate'] * 100)}%, "
        f"betrayals: {lt_context['total_betrayals']}, "
        f"alliances: {lt_context['total_alliances']}."
        if lt_context else "No cross-simulation record."
    )

    recent_history = "\n".join(state.history[-3:]) if state.history else "None yet."
    other_agents = ", ".join(
        f"{k}: {v.name}"
        for k, v in state.countries.items()
        if k not in (sender_id, recipient_id)
    )

    prompt = f"""You are {sender.name}, a strategic negotiator at an international summit.

SCENARIO: {state.scenario}

YOUR PROFILE:
  Goals: {sender.goals}
  Problems: {sender.problems}
  Resources: {sender.resources}
  Trust level: {sender.trust:.2f}
  Current payoff: {sender.payoff:.1f}

YOU ARE WRITING SPECIFICALLY TO: {recipient.name}
  Their goals: {recipient.goals}
  Their problems: {recipient.problems}
  Your history with them this session: {memory_summary}
  Their long-term reputation across past simulation: {lt_summary}

OTHER PARTIES (not this message's recipient): {other_agents}
ACTIVE GAME-THEORY MODELS: {state.game_models}
RECENT HISTORY: {recent_history}

TASK: Write a private diplomatic message TO {recipient.name} specifically.

Your message MUST:
- Be addressed to {recipient.name} and reference their specific situation
- Be under {MESSAGE_CHAR_LIMIT} characters total (this is a hard limit — be concise)
- Reference the actual scenario, not generic platitudes
- Be strategically different from messages you'd send to other parties
- Take ONE of these stances: propose a deal, offer an alliance, issue a warning, or apply pressure
- If {recipient.name} has a poor long-term reputation (low cooperation, high betrayals), reflect that distrust in your tone

STRICT RULES:
- Output ONLY the message text. No labels, headers, signatures, or metadata.
- Do NOT output "C", "D", or any single letter.
- Do NOT start with "Message:" or "Diplomatic message:".
- Make it specific to {recipient.name} — not a generic template.
- Stay under {MESSAGE_CHAR_LIMIT} characters.
"""

    content = invoke_llm(prompt, agent_id=sender_id).strip()

    # Strip any accidental labels the model might prepend
    for prefix in ["Message:", "Diplomatic message:", "Response:", f"{sender.name}:"]:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()

    # Enforce character limit — truncate at last complete sentence if over limit
    if len(content) > MESSAGE_CHAR_LIMIT:
        truncated = content[:MESSAGE_CHAR_LIMIT]
        # Try to cut at the last sentence boundary
        last_period = max(
            truncated.rfind(". "),
            truncated.rfind("! "),
            truncated.rfind("? "),
        )
        if last_period > MESSAGE_CHAR_LIMIT // 2:
            content = truncated[:last_period + 1]
        else:
            content = truncated.rstrip() + "…"

    # Safety check — if still too short or a single letter, make a
    # recipient-SPECIFIC fallback (not a generic one)
    if len(content) < 20 or content.upper() in {"C", "D", "COOPERATE", "DEFECT"}:
        content = _fallback_message(sender, recipient, state.scenario, memory_summary)

    return Message(
        sender=sender_id,
        recipient=recipient_id,
        content=content,
        round=state.round + 1,
    )


def _fallback_message(sender, recipient, scenario: str, memory_summary: str) -> str:
    """
    Generates a recipient-specific fallback message so that even in the
    worst case, each message is unique and references the actual parties.
    This replaces the old generic template that produced identical messages.
    """
    scenario_snippet = scenario[:80] if len(scenario) > 80 else scenario
    coop_hint = "our history of cooperation" if "0." in memory_summary else "the stakes at hand"

    return (
        f"{recipient.name}, given {coop_hint} and the pressure of {scenario_snippet}, "
        f"I urge you to consider a direct agreement with {sender.name}. "
        f"The alternative serves neither of us."
    )[:MESSAGE_CHAR_LIMIT]


def run_messaging_phase(state: NegotiationState) -> NegotiationState:
    """Every agent sends a private, recipient-specific message to every other agent."""
    agent_ids = list(state.countries.keys())

    for agent in state.countries.values():
        agent.inbox = []
        agent.outbox = []

    all_messages = []

    for sender_id in agent_ids:
        for recipient_id in agent_ids:
            if sender_id == recipient_id:
                continue

            msg = compose_message(state, sender_id, recipient_id)
            all_messages.append(msg)
            state.countries[sender_id].outbox.append(msg)
            state.countries[recipient_id].inbox.append(msg)

            print(
                f"\n  📨 {state.countries[sender_id].name} "
                f"→ {state.countries[recipient_id].name} "
                f"({len(msg.content)} chars):"
            )
            print(f"     \"{msg.content}\"")

    state.message_log.extend(all_messages)
    return state


# ---------------------------------------------------------------------------
# PHASE 2: DECISION
# ---------------------------------------------------------------------------

def country_decide(
    state: NegotiationState,
    country_id: str,
    all_opponent_ids: list,
) -> tuple:
    """
    Each agent reads its inbox, considers game-theory strategy,
    and returns (action, proposed_ally_id).
    """
    country = state.countries[country_id]
    opponents = {k: state.countries[k] for k in all_opponent_ids}

    memory_lines = []
    for opp_id, opp in opponents.items():
        mem = country.memory.get(opp_id)
        if mem:
            memory_lines.append(
                f"  {opp.name} ({opp_id}): coop_rate={mem.cooperation_rate:.2f}, "
                f"betrayals={mem.betrayal_count}, alliances={mem.alliance_count}"
            )
        else:
            memory_lines.append(f"  {opp.name} ({opp_id}): no history yet")

    memory_summary = "\n".join(memory_lines)

    inbox_summary = "\n".join(
        f"  From {state.countries[m.sender].name} ({m.sender}): \"{m.content}\""
        for m in country.inbox
    ) if country.inbox else "  (No messages received this round)"

    recent_history = "\n".join(state.history[-5:]) if state.history else "None."

    prompt = f"""You are {country.name} (Agent {country_id}), negotiating at an international summit.

SCENARIO: {state.scenario}

YOUR PROFILE:
  Goals: {country.goals}
  Problems: {country.problems}
  Resources: {country.resources}
  Trust level: {country.trust:.2f}
  Accumulated payoff: {country.payoff:.1f}
  Current coalition partner: {country.current_coalition_partner or "none"}

OPPONENTS:
{chr(10).join(f"  {k}: {v.name} | trust={v.trust:.2f} | payoff={v.payoff:.1f} | their coalition partner: {v.current_coalition_partner or 'none'}" for k, v in opponents.items())}

YOUR MEMORY OF OPPONENTS:
{memory_summary}

MESSAGES YOU RECEIVED THIS ROUND:
{inbox_summary}

NEGOTIATION HISTORY:
{recent_history}

ACTIVE GAME-THEORY MODELS: {state.game_models}

COALITION RULES YOU MUST FOLLOW:
- You can only be in ONE coalition at a time.
- If you propose a new ally while already in a coalition, you EXIT your current one first.
- A coalition only forms if BOTH parties propose each other AND both cooperate.
- Proposing an ally you are already partnered with = maintaining the coalition.

DECISION RULES:
1. REPEATED GAME: Trust > 0.65 → cooperation yields long-term gains.
2. TIT-FOR-TAT: If betrayed last round → consider defecting.
3. COALITION: If excluded by others → ally with a compatible party.
4. DETERRENCE: Dominant payoff → cooperate to maintain position.
5. BRINKMANSHIP: Low payoff + low trust → defect to force recalibration.
6. FREE-RIDER: Defecting while others cooperate → short-term gain, long-term trust loss.
7. NASH BARGAINING: Mutual cooperation maximises joint surplus.

TASK: Return a JSON object ONLY — no explanation, no markdown fences:
{{"action": "C", "proposed_ally": null, "reasoning": "one sentence"}}

action must be exactly "C" or "D".
proposed_ally must be an agent ID ("A", "B", "C", or "D") or null. Not your own ID.
"""

    raw = invoke_llm(prompt, agent_id=country_id).strip()

    action = "C"
    proposed_ally = None

    try:
        import json
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
        parsed = json.loads(clean)
        action = str(parsed.get("action", "C")).upper().strip()
        proposed_ally = parsed.get("proposed_ally")
        reasoning = parsed.get("reasoning", "")

        if action not in {"C", "D"}:
            action = "C"

        # Validate proposed_ally — must be a real agent, not self
        if proposed_ally not in state.countries or proposed_ally == country_id:
            proposed_ally = None

        print(
            f"\n  🤔 {country.name} ({country_id}) → action={action}, "
            f"ally={proposed_ally} | {reasoning}"
        )

    except Exception:
        raw_upper = raw.upper().strip()
        if (raw_upper.startswith("D") or
                '"action": "D"' in raw_upper or
                "'action': 'd'" in raw_upper.lower()):
            action = "D"
        else:
            action = "C" if country.trust >= 0.5 else "D"

    return action, proposed_ally