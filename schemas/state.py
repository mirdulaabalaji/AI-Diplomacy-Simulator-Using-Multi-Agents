from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CountryMemory(BaseModel):
    opponent: str
    cooperation_rate: float = 0.5
    betrayal_count: int = 0
    alliance_count: int = 0


class Message(BaseModel):
    sender: str
    recipient: str
    content: str
    round: int


class CountryState(BaseModel):
    name: str
    resources: List[str]
    problems: List[str]
    goals: List[str]

    trust: float = 0.5
    payoff: float = 0.0

    memory: Dict[str, CountryMemory] = Field(default_factory=dict)
    inbox: List[Message] = Field(default_factory=list)
    outbox: List[Message] = Field(default_factory=list)
    current_coalition_partner: Optional[str] = None
    proposed_ally: Optional[str] = None


class NegotiationState(BaseModel):
    scenario: str
    countries: Dict[str, CountryState]

    round: int = 0
    history: List[str] = Field(default_factory=list)
    message_log: List[Message] = Field(default_factory=list)
    game_models: List[str] = Field(default_factory=list)

    agreement_reached: bool = False
    agreement_failed: bool = False          # NEW: explicit defection ending

    active_coalitions: List[List[str]] = Field(default_factory=list)
    coalition_history: List[dict] = Field(default_factory=list)