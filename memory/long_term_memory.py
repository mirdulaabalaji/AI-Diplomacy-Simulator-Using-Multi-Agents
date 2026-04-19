"""
memory/long_term_memory.py

Cross-simulation persistent memory using SQLite.

Stores:
  - Past simulation summaries (scenario, outcome, agents, payoffs, rounds)
  - Per-agent behavioural history across simulations (cooperation rate, betrayals, alliances)
  - Retrieved context is injected into agent prompts so agents "remember" past encounters

Usage:
  from memory.long_term_memory import SimulationMemory
  mem = SimulationMemory()                  # opens/creates negotiation_memory.db
  mem.save_simulation(final_state, outcome) # call after each simulation
  mem.get_agent_history("Germany")          # returns past stats for a named agent
  mem.get_recent_simulations(n=3)           # returns last n simulation summaries
  mem.get_context_for_prompt(agent_names)   # returns a formatted string for LLM prompts
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# DB lives next to this file (inside the memory/ folder)
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "negotiation_memory.db")


class SimulationMemory:
    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create tables if they don't exist yet."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at      TEXT    NOT NULL,
                    scenario    TEXT    NOT NULL,
                    outcome     TEXT    NOT NULL,   -- FULL_AGREEMENT | COLLAPSED | PARTIAL_AGREEMENT | COALITION_ONLY
                    outcome_label TEXT  NOT NULL,
                    rounds      INTEGER NOT NULL,
                    game_models TEXT    NOT NULL,   -- JSON list
                    agents_json TEXT    NOT NULL    -- JSON dict of agent results
                );

                CREATE TABLE IF NOT EXISTS agent_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    sim_id          INTEGER NOT NULL REFERENCES simulations(id),
                    agent_name      TEXT    NOT NULL,
                    agent_id        TEXT    NOT NULL,
                    final_payoff    REAL    NOT NULL,
                    final_trust     REAL    NOT NULL,
                    coop_rate       REAL    NOT NULL DEFAULT 0.5,
                    betrayal_count  INTEGER NOT NULL DEFAULT 0,
                    alliance_count  INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_stats(agent_name);
                CREATE INDEX IF NOT EXISTS idx_sim_run_at ON simulations(run_at);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────────────────────────────────────

    def save_simulation(self, final_state, outcome: dict) -> int:
        """
        Persist one completed simulation.

        Parameters
        ----------
        final_state : NegotiationState
            The final state object returned by the LangGraph run.
        outcome : dict
            The outcome dict built by _get_outcome() in streamlit_app.py.

        Returns
        -------
        int  — the new simulation row id
        """
        agents_json = {
            k: {
                "name":          v.name,
                "final_payoff":  round(v.payoff, 2),
                "final_trust":   round(v.trust, 2),
                "coop_rate":     round(
                    sum(m.cooperation_rate for m in v.memory.values()) / len(v.memory)
                    if v.memory else 0.5,
                    3,
                ),
                "betrayal_count": sum(m.betrayal_count for m in v.memory.values()),
                "alliance_count": sum(m.alliance_count for m in v.memory.values()),
            }
            for k, v in final_state.countries.items()
        }

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO simulations
                   (run_at, scenario, outcome, outcome_label, rounds, game_models, agents_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    final_state.scenario[:1000],
                    outcome.get("result", "UNKNOWN"),
                    outcome.get("label", ""),
                    final_state.round,
                    json.dumps(final_state.game_models),
                    json.dumps(agents_json),
                ),
            )
            sim_id = cur.lastrowid

            for agent_id, info in agents_json.items():
                conn.execute(
                    """INSERT INTO agent_stats
                       (sim_id, agent_name, agent_id, final_payoff, final_trust,
                        coop_rate, betrayal_count, alliance_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sim_id,
                        info["name"],
                        agent_id,
                        info["final_payoff"],
                        info["final_trust"],
                        info["coop_rate"],
                        info["betrayal_count"],
                        info["alliance_count"],
                    ),
                )

        return sim_id

    # ─────────────────────────────────────────────────────────────────────────
    # READ
    # ─────────────────────────────────────────────────────────────────────────

    def get_recent_simulations(self, n: int = 5) -> List[dict]:
        """Return the n most recent simulation summaries (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM simulations ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_history(self, agent_name: str) -> Optional[dict]:
        """
        Aggregate stats for a named agent across all past simulations.
        Returns None if agent has never appeared.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT
                       COUNT(*)           AS appearances,
                       AVG(final_payoff)  AS avg_payoff,
                       AVG(final_trust)   AS avg_trust,
                       AVG(coop_rate)     AS avg_coop_rate,
                       SUM(betrayal_count) AS total_betrayals,
                       SUM(alliance_count) AS total_alliances
                   FROM agent_stats
                   WHERE LOWER(agent_name) = LOWER(?)""",
                (agent_name,),
            ).fetchone()

        if not rows or rows["appearances"] == 0:
            return None

        return {
            "agent_name":     agent_name,
            "appearances":    rows["appearances"],
            "avg_payoff":     round(rows["avg_payoff"] or 0, 2),
            "avg_trust":      round(rows["avg_trust"] or 0, 2),
            "avg_coop_rate":  round(rows["avg_coop_rate"] or 0, 2),
            "total_betrayals": int(rows["total_betrayals"] or 0),
            "total_alliances": int(rows["total_alliances"] or 0),
        }

    def get_context_for_prompt(self, agent_names: List[str]) -> str:
        """
        Build a compact memory summary string suitable for injection
        into an LLM prompt.

        Example output
        --------------
        [LONG-TERM MEMORY]
        Germany (3 past sims): avg payoff=8.4, avg trust=0.61,
            cooperation rate=72%, betrayals=2, alliances=5
        France (1 past sim): avg payoff=5.0, avg trust=0.45, ...
        Recent outcomes: FULL_AGREEMENT (2025-12-01), COLLAPSED (2025-11-28)
        """
        lines = ["[LONG-TERM MEMORY — cross-simulation context]"]

        for name in agent_names:
            hist = self.get_agent_history(name)
            if hist:
                sim_word = "sim" if hist["appearances"] == 1 else "sims"
                lines.append(
                    f"  {name} ({hist['appearances']} past {sim_word}): "
                    f"avg payoff={hist['avg_payoff']}, "
                    f"avg trust={hist['avg_trust']}, "
                    f"coop rate={int(hist['avg_coop_rate'] * 100)}%, "
                    f"betrayals={hist['total_betrayals']}, "
                    f"alliances={hist['total_alliances']}"
                )

        recent = self.get_recent_simulations(n=3)
        if recent:
            summaries = ", ".join(
                f"{r['outcome']} ({r['run_at'][:10]})" for r in recent
            )
            lines.append(f"  Recent outcomes: {summaries}")

        if len(lines) == 1:
            return ""  # nothing stored yet — return empty so prompts aren't cluttered

        return "\n".join(lines)

    def get_all_agent_names(self) -> List[str]:
        """Return every unique agent name ever seen (for the UI history panel)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT agent_name FROM agent_stats ORDER BY agent_name"
            ).fetchall()
        return [r["agent_name"] for r in rows]

    def count_simulations(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]

    def clear_all(self):
        """Delete every simulation and agent stat record, resetting memory to zero."""
        with self._connect() as conn:
            conn.execute("DELETE FROM agent_stats")
            conn.execute("DELETE FROM simulations")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('simulations', 'agent_stats')")