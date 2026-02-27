from typing import Dict, List
from pydantic import BaseModel, Field


class CountryMemory(BaseModel):
    opponent: str
    cooperation_rate: float = 0.5
    betrayal_count: int = 0


class CountryState(BaseModel):
    name: str
    resources: List[str]
    problems: List[str]
    goals: List[str]

    trust: float = 0.5
    payoff: float = 0.0

    # midterm strategic memory
    memory: Dict[str, CountryMemory] = Field(default_factory=dict)


class NegotiationState(BaseModel):
    scenario: str

    countries: Dict[str, CountryState]

    round: int = 0
    history: List[str] = Field(default_factory=list)

    game_models: List[str] = Field(default_factory=list)

    agreement_reached: bool = False