import streamlit as st
import threading
import time
import json
import sqlite3
from collections import defaultdict, Counter

from utils.stream_bridge import push_event, pop_events, mark_done, is_done, reset, SimEvent
from agents.llm import AGENT_MODELS
from memory.long_term_memory import SimulationMemory

_sim_memory = SimulationMemory()

st.set_page_config(
    page_title="Negotiation Simulator",
    page_icon="🌐",
    layout="wide",
)

AGENT_COLORS = {
    "A": "#4A90D9",
    "B": "#E8563A",
    "C": "#2ECC71",
    "D": "#9B59B6",
}

ACTION_EMOJI = {"C": "🤝", "D": "⚔️"}
ACTION_LABEL = {"C": "Cooperate", "D": "Defect"}

MODEL_DESCRIPTIONS = {
    "Nash Bargaining": "Both parties negotiate toward a solution that maximises joint surplus.",
    "Repeated Game": "Long-term relationships make cooperation more rational than defection.",
    "Prisoner's Dilemma": "Individual rationality leads to collective suboptimality.",
    "Coalition Formation": "Subgroups ally to gain leverage over non-members.",
    "Deterrence Theory": "Credible threats prevent adversaries from defecting.",
    "Brinkmanship Theory": "Pushing to the edge of conflict to force concessions.",
    "Free-Rider Problem": "One party benefits from others' cooperation without contributing.",
    "Escalation Control": "Managing conflict intensity to avoid mutually destructive outcomes.",
    "Information Asymmetry & Signaling": "Parties reveal or conceal information to gain strategic advantage.",
}

# ── session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "running": False,
    "events": [],
    "final_state": None,
    "countries": {},
    "current_round": 0,
    "countries_rendered": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_patched(scenario_text: str):
    import agents.country_agent as ca
    import agents.evaluator as ev
    from agents.input_parser import parse_natural_language_input
    from graph.negotiation_graph import build_negotiation_graph
    from schemas.state import NegotiationState

    original_compose  = ca.compose_message
    original_decide   = ca.country_decide
    original_evaluate = ev.evaluate_round

    def patched_compose(state, sender_id, recipient_id):
        msg = original_compose(state, sender_id, recipient_id)
        push_event(SimEvent(
            type="message",
            round=state.round + 1,
            sender=sender_id,
            recipient=recipient_id,
            content=msg.content,
        ))
        return msg

    def patched_decide(state, country_id, all_opponent_ids):
        action, ally = original_decide(state, country_id, all_opponent_ids)
        push_event(SimEvent(
            type="decision",
            round=state.round + 1,
            sender=country_id,
            action=action,
            proposed_ally=ally,
        ))
        return action, ally

    def patched_evaluate(state, actions, proposed_allies):
        result = original_evaluate(state, actions, proposed_allies)
        push_event(SimEvent(
            type="round_result",
            round=result.round,
            payoffs={k: round(v.payoff, 1) for k, v in result.countries.items()},
            coalitions=result.active_coalitions,
            content=result.history[-1] if result.history else "",
        ))
        return result

    ca.compose_message  = patched_compose
    ca.country_decide   = patched_decide
    ev.evaluate_round   = patched_evaluate

    try:
        push_event(SimEvent(type="system", content="🔍 Parsing scenario..."))
        state = parse_natural_language_input(scenario_text)

        country_profiles = {
            k: {
                "name":      v.name,
                "goals":     v.goals,
                "problems":  v.problems,
                "resources": v.resources,
                "model":     AGENT_MODELS.get(k, "unknown"),
            }
            for k, v in state.countries.items()
        }
        push_event(SimEvent(type="countries_ready", content=json.dumps(country_profiles)))
        push_event(SimEvent(type="system", content=f"📋 Scenario: {state.scenario}"))
        push_event(SimEvent(type="system", content="🚀 Starting negotiation..."))

        graph  = build_negotiation_graph()
        result = graph.invoke(state)

        final_state = NegotiationState(**result) if isinstance(result, dict) else result

        outcome = _get_outcome(final_state)

        outcome["_final_scores"] = {
            k: {"name": v.name, "payoff": round(v.payoff, 2), "trust": round(v.trust, 2)}
            for k, v in final_state.countries.items()
        }
        outcome["_game_models"] = final_state.game_models
        outcome["_history"]     = final_state.history

        push_event(SimEvent(type="outcome", content=outcome["label"], outcome=outcome))

        output = _build_output(final_state, outcome)
        with open("final_output.json", "w") as f:
            json.dump(output, f, indent=2)

        try:
            _sim_memory.save_simulation(final_state, outcome)
        except Exception as mem_err:
            push_event(SimEvent(type="system", content=f"⚠️ Memory save failed: {mem_err}"))

        st.session_state.final_state = final_state

    except Exception as e:
        import traceback
        push_event(SimEvent(type="system", content=f"❌ Error: {str(e)}"))
        push_event(SimEvent(type="system", content=traceback.format_exc()))
    finally:
        ca.compose_message  = original_compose
        ca.country_decide   = original_decide
        ev.evaluate_round   = original_evaluate
        mark_done()


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME + OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_outcome(state) -> dict:
    history = state.history
    if not history:
        return {"result": "NO_ROUNDS", "label": "❌ No rounds occurred.",
                "detail": "", "color": "#888"}

    if state.agreement_reached:
        return {
            "result": "FULL_AGREEMENT",
            "label":  "✅ Full agreement reached",
            "detail": "All four parties cooperated — Nash equilibrium achieved.",
            "color":  "#2ECC71",
        }

    if state.agreement_failed:
        defector_rounds = {}
        for entry in history:
            try:
                ap = entry.split(":")[1].split("|")[0]
                for pair in ap.split(","):
                    k, v = pair.strip().split("=")
                    if v.strip() == "D":
                        defector_rounds[k.strip()] = defector_rounds.get(k.strip(), 0) + 1
            except Exception:
                pass
        defectors = [
            f"{state.countries[k].name} ({k}) defected {n}x"
            for k, n in sorted(defector_rounds.items(), key=lambda x: -x[1])
            if k in state.countries
        ]
        return {
            "result":           "COLLAPSED",
            "label":            "❌ Negotiation collapsed — no agreement reached",
            "detail":           f"Ran all {state.round} rounds without full cooperation.",
            "defectors":        defectors,
            "game_theory_note": (
                "Prisoner's Dilemma trap: individually rational defection "
                "produced a collectively suboptimal outcome."
            ),
            "color": "#E8563A",
        }

    def majority_coop(r):
        try:
            ap = r.split(":")[1].split("|")[0]
            return sum(
                1 for p in ap.split(",")
                if p.strip().split("=")[1].strip() == "C"
            ) >= 3
        except Exception:
            return False

    if sum(1 for r in history[-3:] if majority_coop(r)) >= 2:
        return {
            "result": "PARTIAL_AGREEMENT",
            "label":  "⚠️ Partial agreement — majority cooperation",
            "detail": "3+ parties cooperated consistently but full consensus not reached.",
            "color":  "#F39C12",
        }

    if state.active_coalitions:
        partners = " & ".join(
            f"{state.countries[a].name}+{state.countries[b].name}"
            for a, b in state.active_coalitions
        )
        return {
            "result": "COALITION_ONLY",
            "label":  f"⚠️ Coalition only: {partners}",
            "detail": "A bilateral alliance formed but no full group agreement.",
            "color":  "#F39C12",
        }

    return {
        "result":           "COLLAPSED",
        "label":            "❌ Negotiation collapsed",
        "detail":           "No stable equilibrium reached within the round limit.",
        "game_theory_note": "Defection dominated all rounds.",
        "color":            "#E8563A",
    }


def _build_output(final_state, outcome) -> dict:
    return {
        "scenario": final_state.scenario,
        "outcome":  outcome,
        "agents": {
            k: {
                "name":          v.name,
                "model":         AGENT_MODELS.get(k, "unknown"),
                "final_payoff":  round(v.payoff, 2),
                "final_trust":   round(v.trust, 2),
                "proposed_ally": v.proposed_ally,
            }
            for k, v in final_state.countries.items()
        },
        "game_theory_models_used": final_state.game_models,
        "negotiation_rounds":      final_state.round,
        "active_coalitions":       final_state.active_coalitions,
        "negotiation_history":     final_state.history,
        "message_log": [
            {
                "round":   m.round,
                "from":    final_state.countries[m.sender].name,
                "to":      final_state.countries[m.recipient].name,
                "message": m.content,
            }
            for m in final_state.message_log
        ],
        "agreement_reached": final_state.agreement_reached,
        "agreement_failed":  final_state.agreement_failed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_country_cards(countries: dict):
    st.markdown("### 🌍 Negotiating Parties")
    cols = st.columns(4)
    for i, (agent_id, info) in enumerate(countries.items()):
        color = AGENT_COLORS.get(agent_id, "#888")
        with cols[i]:
            st.markdown(
                f"""<div style="border:2px solid {color};border-radius:14px;
                padding:14px;background:{color}10;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div style="background:{color};color:white;border-radius:50%;
                        width:36px;height:36px;display:flex;align-items:center;
                        justify-content:center;font-weight:800;font-size:18px;
                        flex-shrink:0;">{agent_id}</div>
                    <div>
                        <div style="font-weight:700;font-size:15px;">{info["name"]}</div>
                        <div style="font-size:11px;color:#888;font-family:monospace;">
                            🤖 {info["model"]}</div>
                    </div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{color};"
                f"text-transform:uppercase;letter-spacing:0.8px;"
                f"margin-bottom:4px;'>🎯 Goals</div>",
                unsafe_allow_html=True,
            )
            for g in (info["goals"] or ["—"]):
                st.markdown(f"- {g}")
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{color};"
                f"text-transform:uppercase;letter-spacing:0.8px;"
                f"margin-top:8px;margin-bottom:4px;'>⚠️ Problems</div>",
                unsafe_allow_html=True,
            )
            for p in (info["problems"] or ["—"]):
                st.markdown(f"- {p}")
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{color};"
                f"text-transform:uppercase;letter-spacing:0.8px;"
                f"margin-top:8px;margin-bottom:6px;'>💎 Resources</div>",
                unsafe_allow_html=True,
            )
            if info["resources"]:
                st.markdown(" ".join(f"`{r}`" for r in info["resources"]))
            else:
                st.markdown("—")
            st.markdown("---")

    st.markdown("""
    <div style="text-align:center;color:#555;font-size:13px;
        padding-top:4px;margin-bottom:8px;">
        ⬇️ Negotiation begins below
    </div>
    """, unsafe_allow_html=True)


def render_round_header(round_num: int):
    st.markdown(f"""
    <div style="background:#1a1a2e;border-radius:10px;padding:10px 18px;
        margin:20px 0 10px 0;border-left:5px solid #4A90D9;">
        <span style="font-size:20px;font-weight:700;color:#4A90D9;">
            Round {round_num}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_message_bubble(event: SimEvent, countries: dict):
    sender_name = countries.get(event.sender, {}).get("name", event.sender)
    recip_name  = countries.get(event.recipient, {}).get("name", event.recipient)
    color = AGENT_COLORS.get(event.sender, "#888")
    st.markdown(f"""
    <div style="border-left:4px solid {color};background:{color}12;
        border-radius:0 10px 10px 0;padding:10px 14px;margin:6px 0;">
        <div style="font-size:12px;color:#888;margin-bottom:4px;">
            📨 <b style="color:{color}">{sender_name}</b> → <b>{recip_name}</b>
        </div>
        <div style="font-size:14px;line-height:1.6;">{event.content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_decisions(dec_events: list, countries: dict):
    st.markdown("**⚖️ Decisions this round:**")
    dec_html = ""
    for ev in dec_events:
        name   = countries.get(ev.sender, {}).get("name", ev.sender)
        color  = AGENT_COLORS.get(ev.sender, "#888")
        action = ev.action or "?"
        emoji  = ACTION_EMOJI.get(action, "❓")
        label  = ACTION_LABEL.get(action, action)
        ally_str = ""
        if ev.proposed_ally:
            ally_name = countries.get(ev.proposed_ally, {}).get("name", ev.proposed_ally)
            ally_str = f" · 🤜 → {ally_name}"
        bg     = "#2ECC7120" if action == "C" else "#E8563A20"
        border = "#2ECC71"   if action == "C" else "#E8563A"
        dec_html += (
            f'<div style="display:inline-block;border:1.5px solid {border};'
            f'background:{bg};border-radius:20px;padding:5px 16px;'
            f'margin:4px;font-size:13px;">'
            f'<b style="color:{color}">{name}</b>: {emoji} {label}{ally_str}'
            f'</div>'
        )
    st.markdown(dec_html, unsafe_allow_html=True)


def render_round_payoffs(payoffs: dict, countries: dict):
    st.markdown("**📊 Payoffs after this round:**")
    cols = st.columns(len(payoffs))
    for i, (agent_id, payoff) in enumerate(payoffs.items()):
        name  = countries.get(agent_id, {}).get("name", agent_id)
        color = AGENT_COLORS.get(agent_id, "#888")
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:8px;border:1px solid {color}44;
                border-radius:10px;background:{color}10;">
                <div style="font-size:11px;color:{color};font-weight:600;
                    margin-bottom:4px;">{name}</div>
                <div style="font-size:26px;font-weight:800;color:{color};">{payoff}</div>
                <div style="font-size:10px;color:#666;">payoff</div>
            </div>
            """, unsafe_allow_html=True)


def render_outcome_banner(outcome: dict, final_state=None):
    color  = outcome.get("color", "#888")
    label  = outcome.get("label", "")
    detail = outcome.get("detail", "")

    st.markdown(f"""
    <div style="border:3px solid {color};border-radius:16px;padding:24px;
        background:{color}18;margin:24px 0;text-align:center;">
        <div style="font-size:28px;font-weight:800;color:{color};">{label}</div>
        <div style="font-size:15px;margin-top:8px;color:#ccc;">{detail}</div>
    </div>
    """, unsafe_allow_html=True)

    defectors = outcome.get("defectors", [])
    if defectors:
        st.markdown(
            "<div style='font-size:13px;color:#aaa;margin-bottom:6px;"
            "font-weight:600;'>🚨 Persistent defectors:</div>",
            unsafe_allow_html=True,
        )
        for d in defectors:
            st.markdown(
                f"<div style='background:#2a1010;border-left:3px solid #E8563A;"
                f"border-radius:0 8px 8px 0;padding:6px 12px;margin:4px 0;"
                f"font-size:14px;color:#ddd;'>⚔️ {d}</div>",
                unsafe_allow_html=True,
            )

    note = outcome.get("game_theory_note", "")
    if note:
        st.markdown(
            f"<div style='margin-top:12px;font-size:13px;color:#999;"
            f"font-style:italic;padding:10px 14px;background:#1a1a1a;"
            f"border-radius:8px;border-left:3px solid {color}88;'>"
            f"📖 {note}</div>",
            unsafe_allow_html=True,
        )

    scores      = outcome.get("_final_scores")
    game_models = outcome.get("_game_models", [])
    history     = outcome.get("_history", [])

    if not scores and final_state:
        scores = {
            k: {"name": v.name, "payoff": round(v.payoff, 2), "trust": round(v.trust, 2)}
            for k, v in final_state.countries.items()
        }
    if not game_models and final_state:
        game_models = final_state.game_models
    if not history and final_state:
        history = final_state.history

    if scores:
        st.markdown("### 📊 Final Scores")
        all_payoffs = [s["payoff"] for s in scores.values()]
        max_payoff  = max(all_payoffs) if max(all_payoffs) > 0 else 1
        cols = st.columns(len(scores))
        for i, (agent_id, s) in enumerate(scores.items()):
            agent_color = AGENT_COLORS.get(agent_id, "#888")
            payoff_pct  = int((s["payoff"] / max_payoff) * 100)
            trust_pct   = int(s["trust"] * 100)
            is_winner   = s["payoff"] == max(all_payoffs)
            crown       = "👑 " if is_winner else ""
            trust_color = (
                "#2ECC71" if s["trust"] >= 0.6
                else "#F39C12" if s["trust"] >= 0.35
                else "#E8563A"
            )
            with cols[i]:
                st.markdown(f"""
                <div style="border:1.5px solid {agent_color}55;border-radius:12px;
                    padding:16px;background:{agent_color}10;text-align:center;">
                    <div style="display:flex;align-items:center;justify-content:center;
                        gap:8px;margin-bottom:12px;">
                        <div style="background:{agent_color};color:white;
                            border-radius:50%;width:34px;height:34px;
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:17px;">{agent_id}</div>
                        <div style="font-weight:700;font-size:14px;text-align:left;">
                            {crown}{s["name"]}</div>
                    </div>
                    <div style="margin-bottom:14px;">
                        <div style="font-size:11px;color:#888;margin-bottom:4px;
                            text-transform:uppercase;letter-spacing:0.5px;">
                            💰 Payoff</div>
                        <div style="font-size:30px;font-weight:800;
                            color:{agent_color};">{s["payoff"]}</div>
                        <div style="background:#2a2a2a;border-radius:4px;
                            height:7px;margin-top:8px;">
                            <div style="background:{agent_color};width:{payoff_pct}%;
                                height:7px;border-radius:4px;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#888;margin-bottom:4px;
                            text-transform:uppercase;letter-spacing:0.5px;">
                            🤝 Trust</div>
                        <div style="font-size:24px;font-weight:700;
                            color:{trust_color};">{trust_pct}%</div>
                        <div style="background:#2a2a2a;border-radius:4px;
                            height:7px;margin-top:8px;">
                            <div style="background:{trust_color};width:{trust_pct}%;
                                height:7px;border-radius:4px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if game_models:
        st.markdown("### 🧠 Game Theory Models Applied")
        for row_start in range(0, len(game_models), 3):
            row  = game_models[row_start:row_start + 3]
            cols = st.columns(len(row))
            for j, model in enumerate(row):
                desc = MODEL_DESCRIPTIONS.get(
                    model, "A game-theoretic framework applied to this negotiation."
                )
                with cols[j]:
                    st.markdown(f"""
                    <div style="border:1px solid #333;border-radius:10px;
                        padding:14px;background:#1a1a2e;margin-bottom:8px;">
                        <div style="font-size:13px;font-weight:700;
                            color:#4A90D9;margin-bottom:6px;">⚖️ {model}</div>
                        <div style="font-size:12px;color:#aaa;
                            line-height:1.6;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    if history:
        st.markdown("### 📜 Round-by-Round Summary")
        for entry in history:
            has_defection = "=D" in entry
            has_coalition = "Coalitions:" in entry
            if not has_defection:
                icon, bg, border_c = "🤝", "#0d1f0d", "#2ECC7144"
            elif has_coalition:
                icon, bg, border_c = "⚡", "#1f1a0d", "#F39C1244"
            else:
                icon, bg, border_c = "⚔️", "#1f0d0d", "#E8563A44"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border_c};
                border-radius:8px;padding:9px 14px;margin:4px 0;
                font-family:monospace;font-size:13px;color:#ddd;">
                {icon} {entry}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════════════════════════

def _hbar(label, val_str, pct, color):
    """Render a single labelled horizontal bar."""
    w = max(int(pct), 1)
    return (
        f"<div style='margin-bottom:11px;'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:12px;margin-bottom:3px;'>"
        f"<span style='color:#ccc;'>{label}</span>"
        f"<span style='color:{color};font-weight:700;'>{val_str}</span></div>"
        f"<div style='background:#2a2a2a;border-radius:4px;height:9px;'>"
        f"<div style='background:{color};width:{w}%;height:9px;"
        f"border-radius:4px;'></div></div></div>"
    )


def render_analytics():
    total = _sim_memory.count_simulations()
    if total == 0:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#555;">
            <div style="font-size:52px;margin-bottom:16px;">📊</div>
            <div style="font-size:20px;font-weight:600;margin-bottom:8px;">No data yet</div>
            <div style="font-size:14px;">Run at least one simulation to see analytics.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    sims = _sim_memory.get_recent_simulations(n=500)

    with sqlite3.connect(_sim_memory.db_path) as conn:
        conn.row_factory = sqlite3.Row
        agent_rows = [dict(r) for r in conn.execute(
            "SELECT s.id, s.run_at, s.outcome, s.rounds, s.game_models, "
            "       a.agent_name, a.final_payoff, a.final_trust, "
            "       a.coop_rate, a.betrayal_count, a.alliance_count "
            "FROM simulations s JOIN agent_stats a ON a.sim_id = s.id "
            "ORDER BY s.id"
        ).fetchall()]

    outcomes   = [s["outcome"] for s in sims]
    n_agree    = outcomes.count("FULL_AGREEMENT")
    n_partial  = outcomes.count("PARTIAL_AGREEMENT") + outcomes.count("COALITION_ONLY")
    n_collapse = outcomes.count("COLLAPSED")
    avg_rounds = round(sum(s["rounds"] for s in sims) / len(sims), 1)
    agree_rate = round(n_agree / len(sims) * 100) if sims else 0

    # ── KPI strip ─────────────────────────────────────────────────────────────
    def kpi(col, label, value, color):
        col.markdown(
            f"<div style='border:1px solid {color}44;border-radius:12px;"
            f"padding:16px 10px;background:{color}0d;text-align:center;"
            f"margin-bottom:4px;'>"
            f"<div style='font-size:10px;color:#666;text-transform:uppercase;"
            f"letter-spacing:0.8px;margin-bottom:6px;'>{label}</div>"
            f"<div style='font-size:26px;font-weight:800;color:{color};'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Total Simulations",   total,      "#4A90D9")
    kpi(k2, "Full Agreements",     n_agree,    "#2ECC71")
    kpi(k3, "Partial / Coalition", n_partial,  "#F39C12")
    kpi(k4, "Collapsed",           n_collapse, "#E8563A")
    kpi(k5, "Avg Rounds",          avg_rounds, "#9B59B6")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: outcome distribution  +  rounds timeline ───────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🥧 Outcome Distribution")
        total_v = len(sims) or 1
        for lbl, val, col in [
            ("✅ Full Agreement",      n_agree,    "#2ECC71"),
            # ("⚠️ Partial / Coalition", n_partial,  "#F39C12"),
            ("❌ Collapsed",           n_collapse, "#E8563A"),
        ]:
            pct = val / total_v * 100
            st.markdown(_hbar(lbl, f"{val}  ({int(pct)}%)", pct, col),
                        unsafe_allow_html=True)

        rate_col = ("#2ECC71" if agree_rate >= 50
                    else "#F39C12" if agree_rate >= 25 else "#E8563A")
        st.markdown(
            f"<div style='margin-top:12px;padding:12px 16px;border-radius:10px;"
            f"border:1px solid {rate_col}44;background:{rate_col}0d;text-align:center;'>"
            f"<span style='font-size:12px;color:#888;'>Agreement rate  </span>"
            f"<span style='font-size:22px;font-weight:800;color:{rate_col};'>{agree_rate}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("#### 📈 Rounds per Simulation (timeline)")
        timeline = list(reversed(sims))
        if len(timeline) >= 2:
            W, H, PAD = 400, 150, 28
            max_r = max(t["rounds"] for t in timeline) or 1
            min_r = min(t["rounds"] for t in timeline)
            n = len(timeline)
            pts = []
            for i, t in enumerate(timeline):
                x = PAD + (i / (n - 1)) * (W - 2 * PAD)
                y = H - PAD - ((t["rounds"] - min_r) / max(max_r - min_r, 1)) * (H - 2 * PAD)
                pts.append((x, y, t))
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
            dots = "".join(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                f'fill="{"#2ECC71" if t["outcome"] != "COLLAPSED" else "#E8563A"}" '
                f'stroke="#111" stroke-width="1.5"/>'
                for x, y, t in pts
            )
            svg = (
                f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
                f'style="width:100%;background:#111;border-radius:10px;">'
                f'<polyline points="{polyline}" fill="none" stroke="#4A90D966" stroke-width="2"/>'
                f'{dots}'
                f'<text x="{PAD}" y="{H-4}" fill="#444" font-size="10">Sim 1</text>'
                f'<text x="{W-PAD-32}" y="{H-4}" fill="#444" font-size="10">Latest</text>'
                f'<text x="4" y="{PAD+4}" fill="#444" font-size="10">{max_r}r</text>'
                f'<text x="4" y="{H-PAD}" fill="#444" font-size="10">{min_r}r</text>'
                f'<text x="{W//2-80}" y="14" fill="#555" font-size="10">'
                f'Rounds · 🟢 cooperative  🔴 collapsed</text>'
                f'</svg>'
            )
            st.markdown(svg, unsafe_allow_html=True)
        else:
            st.info("Run 2+ simulations to see the timeline.")

    st.markdown("---")

    # ── Row 2: leaderboard  +  coop vs betrayal ───────────────────────────────
    col_c, col_d = st.columns(2)

    agg = defaultdict(lambda: {"payoffs": [], "trusts": [], "coops": [],
                                "betrayals": 0, "alliances": 0})
    for r in agent_rows:
        n = r["agent_name"]
        agg[n]["payoffs"].append(r["final_payoff"])
        agg[n]["trusts"].append(r["final_trust"])
        agg[n]["coops"].append(r["coop_rate"])
        agg[n]["betrayals"] += r["betrayal_count"]
        agg[n]["alliances"] += r["alliance_count"]

    board = sorted(
        [{
            "name":       name,
            "avg_payoff": round(sum(d["payoffs"]) / len(d["payoffs"]), 2),
            "avg_trust":  round(sum(d["trusts"])  / len(d["trusts"]),  2),
            "avg_coop":   round(sum(d["coops"])   / len(d["coops"]),   2),
            "betrayals":  d["betrayals"],
            "alliances":  d["alliances"],
            "sims":       len(d["payoffs"]),
        } for name, d in agg.items()],
        key=lambda x: -x["avg_payoff"],
    )

    with col_c:
        st.markdown("#### 🏆 Agent Leaderboard — Avg Payoff")
        max_p = max((b["avg_payoff"] for b in board), default=1) or 1
        for rank, b in enumerate(board, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            w = max(int(b["avg_payoff"] / max_p * 100), 2)
            trust_col = ("#2ECC71" if b["avg_trust"] >= 0.6
                         else "#F39C12" if b["avg_trust"] >= 0.35 else "#E8563A")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:8px 10px;margin-bottom:6px;border:1px solid #2a2a2a;"
                f"border-radius:10px;background:#111;'>"
                f"<div style='font-size:18px;width:28px;text-align:center;'>{medal}</div>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:13px;font-weight:700;margin-bottom:4px;'>"
                f"{b['name']}"
                f"<span style='font-size:10px;color:#444;font-weight:400;"
                f"margin-left:6px;'>{b['sims']} sim(s)</span></div>"
                f"<div style='background:#222;border-radius:3px;height:6px;'>"
                f"<div style='background:#4A90D9;width:{w}%;height:6px;"
                f"border-radius:3px;'></div></div></div>"
                f"<div style='text-align:right;min-width:80px;'>"
                f"<div style='font-size:16px;font-weight:800;color:#4A90D9;'>"
                f"{b['avg_payoff']}</div>"
                f"<div style='font-size:10px;color:{trust_col};'>"
                f"trust {int(b['avg_trust']*100)}%</div></div></div>",
                unsafe_allow_html=True,
            )

    with col_d:
        st.markdown("#### 🤝 Cooperation vs Betrayal per Agent")
        max_act = max((b["alliances"] + b["betrayals"] for b in board), default=1) or 1
        for b in board:
            coop_w   = int(b["alliances"] / max_act * 200)
            betray_w = int(b["betrayals"] / max_act * 200)
            coop_pct = int(b["avg_coop"] * 100)
            st.markdown(
                f"<div style='margin-bottom:10px;padding:8px 12px;"
                f"border:1px solid #1e1e1e;border-radius:8px;background:#0e0e0e;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:12px;margin-bottom:6px;'>"
                f"<span style='font-weight:600;'>{b['name']}</span>"
                f"<span style='color:#555;'>coop rate {coop_pct}%</span></div>"
                f"<div style='display:flex;gap:3px;align-items:center;margin-bottom:4px;'>"
                f"<div style='background:#2ECC71;height:8px;width:{max(coop_w,2)}px;"
                f"border-radius:3px;'></div>"
                f"<div style='background:#E8563A;height:8px;width:{max(betray_w,2)}px;"
                f"border-radius:3px;'></div></div>"
                f"<div style='display:flex;gap:14px;font-size:10px;color:#444;'>"
                f"<span>🟢 {b['alliances']} alliances</span>"
                f"<span>🔴 {b['betrayals']} betrayals</span></div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Row 3: model frequency  +  rounds distribution ────────────────────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown("#### 🧠 Game Theory Model Frequency")
        model_counts = defaultdict(int)
        for s in sims:
            for m in json.loads(s["game_models"]):
                model_counts[m] += 1
        if model_counts:
            sorted_models = sorted(model_counts.items(), key=lambda x: -x[1])
            max_mc = sorted_models[0][1] or 1
            for model, count in sorted_models:
                st.markdown(
                    _hbar(f"⚖️ {model}", f"{count}x", count / max_mc * 100, "#9B59B6"),
                    unsafe_allow_html=True,
                )

    with col_f:
        st.markdown("#### ⏱️ Rounds Distribution")
        rounds_list = [s["rounds"] for s in sims]
        if rounds_list:
            counts  = Counter(rounds_list)
            max_cnt = max(counts.values()) or 1
            for r in range(min(rounds_list), max(rounds_list) + 1):
                c = counts.get(r, 0)
                out_for_r = [s["outcome"] for s in sims if s["rounds"] == r]
                coop_cnt  = sum(1 for o in out_for_r
                                if o in ("FULL_AGREEMENT", "PARTIAL_AGREEMENT", "COALITION_ONLY"))
                bar_col = ("#2ECC71" if (c > 0 and coop_cnt > len(out_for_r) / 2)
                           else "#E8563A" if c > 0 else "#333")
                st.markdown(
                    _hbar(f'{r} round{"s" if r != 1 else ""}',
                          str(c), c / max_cnt * 100, bar_col),
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Agent trust trajectory SVG line chart ─────────────────────────────────
    st.markdown("#### 📉 Agent Trust Over Simulations")
    st.markdown(
        "<div style='font-size:12px;color:#555;margin-bottom:12px;'>"
        "Final trust score per agent across consecutive simulations.</div>",
        unsafe_allow_html=True,
    )

    trust_by_agent = defaultdict(list)
    for r in agent_rows:
        trust_by_agent[r["agent_name"]].append((r["id"], r["final_trust"]))

    AGENT_COLOURS = {
        "Country A": "#3399FF",
        "Country B": "#FF6600",
        "Country C": "#00DD55",
        "Country D": "#CC44FF",
    }
    # Distinct dash patterns per agent so lines are distinguishable
    # even when trust values are close — this is the key fix for B/D overlap
    AGENT_DASH = {
        "Country A": "none",
        "Country B": "6,3",
        "Country C": "none",
        "Country D": "2,4",
    }
    FALLBACK_PALETTE = ["#F39C12", "#1ABC9C", "#E74C3C", "#3498DB"]
    FALLBACK_DASH    = ["8,4", "4,2", "6,2", "none"]

    if trust_by_agent:
        W2, H2, PAD2 = 680, 190, 40
        all_ids = sorted({r["id"] for r in agent_rows})
        n_ids = max(len(all_ids) - 1, 1)
        id_to_x = {sid: PAD2 + i / n_ids * (W2 - 2 * PAD2)
                   for i, sid in enumerate(all_ids)}

        svg_parts = [
            f'<svg viewBox="0 0 {W2} {H2}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;background:#111;border-radius:10px;">'
        ]
        for tv in [0.25, 0.5, 0.75]:
            gy = H2 - PAD2 - tv * (H2 - 2 * PAD2)
            svg_parts.append(
                f'<line x1="{PAD2}" y1="{gy:.1f}" x2="{W2 - PAD2}" y2="{gy:.1f}" '
                f'stroke="#2a2a2a" stroke-width="1"/>'
                f'<text x="4" y="{gy + 4:.1f}" fill="#555" font-size="9">{tv}</text>'
            )

        # Sort agents by their average trust ascending so the highest-trust
        # agent is drawn last (on top) — prevents any line being buried under another
        def avg_trust(item):
            _, series = item
            vals = [t for _, t in series]
            return sum(vals) / len(vals) if vals else 0

        sorted_agents = sorted(trust_by_agent.items(), key=avg_trust)

        legend_x = PAD2
        fallback_i = 0
        for agent_name, series in sorted_agents:
            ac   = AGENT_COLOURS.get(agent_name)
            dash = AGENT_DASH.get(agent_name, "none")
            if ac is None:
                ac   = FALLBACK_PALETTE[fallback_i % len(FALLBACK_PALETTE)]
                dash = FALLBACK_DASH[fallback_i % len(FALLBACK_DASH)]
                fallback_i += 1

            series_sorted = sorted(series, key=lambda x: x[0])
            pts2 = [(id_to_x.get(sid, PAD2),
                     H2 - PAD2 - trust * (H2 - 2 * PAD2))
                    for sid, trust in series_sorted]

            if len(pts2) >= 2:
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts2)
                dash_attr = f'stroke-dasharray="{dash}"' if dash != "none" else ""
                svg_parts.append(
                    f'<polyline points="{poly}" fill="none" '
                    f'stroke="{ac}" stroke-width="3" opacity="1" {dash_attr}/>'
                )
            # Draw circles last so dots always appear on top of lines
            for x, y in pts2:
                svg_parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                    f'fill="{ac}" stroke="#111" stroke-width="2"/>'
                )
            svg_parts.append(
                f'<rect x="{legend_x}" y="{H2 - 14}" width="14" height="7" '
                f'fill="{ac}" rx="2"/>'
                f'<text x="{legend_x + 18}" y="{H2 - 7}" fill="{ac}" font-size="10">'
                f'{agent_name}</text>'
            )
            legend_x += max(90, len(agent_name) * 7 + 24)

        svg_parts.append("</svg>")
        st.markdown("".join(svg_parts), unsafe_allow_html=True)

    st.markdown("---")

    # ── Raw simulation log ─────────────────────────────────────────────────────
    with st.expander("🗂️ Full Simulation Log", expanded=False):
        for s in sims:
            agents_info = json.loads(s["agents_json"])
            agent_str   = "  ·  ".join(
                f"{info['name']} ({info['final_payoff']}p)"
                for info in agents_info.values()
            )
            oc = {"FULL_AGREEMENT": "#2ECC71", "PARTIAL_AGREEMENT": "#F39C12",
                  "COALITION_ONLY": "#F39C12", "COLLAPSED": "#E8563A"}.get(s["outcome"], "#888")
            st.markdown(
                f"<div style='font-size:12px;padding:6px 10px;margin:3px 0;"
                f"border-left:3px solid {oc};background:{oc}0d;"
                f"border-radius:0 6px 6px 0;font-family:monospace;'>"
                f"<span style='color:{oc};font-weight:700;'>#{s['id']} {s['outcome']}</span>"
                f"  <span style='color:#444;'>{s['run_at'][:16]} · {s['rounds']}r</span><br>"
                f"<span style='color:#555;'>{agent_str}</span></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align:center;margin-bottom:4px;'>🌐 International Negotiation Simulator</h1>
<p style='text-align:center;color:#888;margin-bottom:20px;'>
    4 LLM agents · Game theory · Groq-powered · Real-time messaging
</p>
""", unsafe_allow_html=True)

tab_sim, tab_analytics = st.tabs(["🌐  Simulation", "📊  Analytics"])

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scenario Input")
    scenario_text = st.text_area(
        "Describe the negotiation scenario:",
        height=220,
        placeholder=(
            "e.g. Country A is an oil superpower seeking export profits. "
            "Country B is an industrialized economy dependent on energy imports..."
        ),
        key="scenario_input",
    )

    run_btn = st.button(
        "▶ Run Simulation",
        disabled=st.session_state.running,
        use_container_width=True,
        type="primary",
    )

    if st.button("🔄 Reset", use_container_width=True):
        reset()
        st.session_state.running            = False
        st.session_state.events             = []
        st.session_state.final_state        = None
        st.session_state.countries          = {}
        st.session_state.current_round      = 0
        st.session_state.countries_rendered = False
        st.rerun()

    if st.button("🗑️ Clear Long-Term Memory", use_container_width=True):
        _sim_memory.clear_all()
        st.success("Long-term memory cleared.")
        st.rerun()

    if st.session_state.countries:
        st.divider()
        st.markdown("### 🤖 Active Agents")
        for agent_id, info in st.session_state.countries.items():
            color = AGENT_COLORS.get(agent_id, "#888")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:8px;"
                f"margin-bottom:6px;border:1px solid {color}44;border-radius:8px;"
                f"background:{color}10;'>"
                f"<div style='background:{color};color:white;border-radius:50%;"
                f"width:28px;height:28px;display:flex;align-items:center;"
                f"justify-content:center;font-weight:800;flex-shrink:0;'>"
                f"{agent_id}</div>"
                f"<div style='font-size:13px;font-weight:600;'>{info['name']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── simulation tab ─────────────────────────────────────────────────────────────
with tab_sim:
    if run_btn and scenario_text.strip():
        reset()
        st.session_state.running            = True
        st.session_state.events             = []
        st.session_state.final_state        = None
        st.session_state.countries          = {}
        st.session_state.current_round      = 0
        st.session_state.countries_rendered = False

        thread = threading.Thread(
            target=run_simulation_patched,
            args=(scenario_text,),
            daemon=True,
        )
        thread.start()

    if st.session_state.running or st.session_state.events:
        new_events = pop_events()
        for ev in new_events:
            if ev.type == "countries_ready":
                try:
                    st.session_state.countries = json.loads(ev.content)
                except Exception:
                    pass
            st.session_state.events.append(ev)

        for ev in st.session_state.events:
            if ev.type == "system":
                st.info(ev.content)

        if st.session_state.countries:
            render_country_cards(st.session_state.countries)

        events_by_round: dict = {}
        outcome_event = None

        for ev in st.session_state.events:
            if ev.type in ("system", "countries_ready"):
                continue
            if ev.type == "outcome":
                outcome_event = ev
                continue
            events_by_round.setdefault(ev.round, []).append(ev)

        for round_num in sorted(events_by_round.keys()):
            round_events = events_by_round[round_num]
            render_round_header(round_num)

            msg_events = [e for e in round_events if e.type == "message"]
            if msg_events:
                with st.expander(
                    f"📬 Diplomatic Messages — Round {round_num}", expanded=True
                ):
                    senders = list(dict.fromkeys(e.sender for e in msg_events))
                    cols = st.columns(min(len(senders), 4))
                    for i, sender_id in enumerate(senders):
                        with cols[i % 4]:
                            for ev in [e for e in msg_events if e.sender == sender_id]:
                                render_message_bubble(ev, st.session_state.countries)

            dec_events = [e for e in round_events if e.type == "decision"]
            if dec_events:
                render_decisions(dec_events, st.session_state.countries)

            res_events = [e for e in round_events if e.type == "round_result"]
            for ev in res_events:
                if ev.payoffs:
                    render_round_payoffs(ev.payoffs, st.session_state.countries)
                if ev.coalitions:
                    for pair in ev.coalitions:
                        a_name = st.session_state.countries.get(pair[0], {}).get("name", pair[0])
                        b_name = st.session_state.countries.get(pair[1], {}).get("name", pair[1])
                        st.success(f"🤝 Coalition active: **{a_name}** ↔ **{b_name}**")

            st.markdown("---")

        if outcome_event and outcome_event.outcome:
            render_outcome_banner(outcome_event.outcome, st.session_state.final_state)

        if st.session_state.running and not is_done():
            time.sleep(0.8)
            st.rerun()
        elif is_done():
            st.session_state.running = False

# ── analytics tab ──────────────────────────────────────────────────────────────
with tab_analytics:
    render_analytics()