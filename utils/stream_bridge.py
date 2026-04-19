"""
stream_bridge.py

Captures simulation events (messages, decisions, round results)
as they happen and makes them available to Streamlit in real time.
Instead of printing to terminal, the simulation pushes events into
a shared queue that the Streamlit UI reads from.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimEvent:
    """A single simulation event pushed to the UI."""
    type: str          # "message", "decision", "round_result", "system", "outcome"
    round: int = 0
    sender: Optional[str] = None
    recipient: Optional[str] = None
    content: str = ""
    action: Optional[str] = None       # "C" or "D"
    proposed_ally: Optional[str] = None
    reasoning: Optional[str] = None
    payoffs: Optional[dict] = None
    coalitions: Optional[list] = None
    outcome: Optional[dict] = None


# Global event queue — simulation pushes here, Streamlit reads from here
_event_queue: queue.Queue = queue.Queue()
_simulation_done = threading.Event()


def push_event(event: SimEvent):
    _event_queue.put(event)


def pop_events() -> list[SimEvent]:
    events = []
    while not _event_queue.empty():
        try:
            events.append(_event_queue.get_nowait())
        except queue.Empty:
            break
    return events


def mark_done():
    _simulation_done.set()


def is_done() -> bool:
    return _simulation_done.is_set()


def reset():
    global _event_queue, _simulation_done
    _event_queue = queue.Queue()
    _simulation_done = threading.Event()