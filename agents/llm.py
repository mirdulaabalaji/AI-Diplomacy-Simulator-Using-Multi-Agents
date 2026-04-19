import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# Four distinct Groq models — one per negotiating agent.
# Each model has a different reasoning style, creating genuine
# diversity in negotiation behaviour (like outsmart's multi-LLM arena).

AGENT_MODELS = {
    "A": "llama-3.1-8b-instant",      # large, strategic thinker
    "B": "llama-3.3-70b-versatile",          # fast, opportunistic
    "C": "llama-3.1-8b-instant",                  # Google Gemma — cautious/diplomatic
    "D": "llama-3.3-70b-versatile",            # fast, opportunistic
}

_llm_cache: dict = {}


def get_llm(agent_id: str) -> ChatGroq:
    """
    Returns a cached ChatGroq instance for the given agent.
    Each agent gets its own model so behaviour genuinely differs.
    """
    if agent_id not in _llm_cache:
        model_name = AGENT_MODELS.get(agent_id, "llama-3.1-8b-instant")
        _llm_cache[agent_id] = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            temperature=0.7,
            timeout=45,
        )
    return _llm_cache[agent_id]


def invoke_llm(prompt: str, agent_id: str = "A") -> str:
    """
    Invokes the Groq LLM assigned to the given agent and returns the response.
    Falls back gracefully if the model call fails.
    """
    llm = get_llm(agent_id)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] Agent {agent_id} ({AGENT_MODELS.get(agent_id)}): {e}")
        return "C"   # safe fallback: cooperate