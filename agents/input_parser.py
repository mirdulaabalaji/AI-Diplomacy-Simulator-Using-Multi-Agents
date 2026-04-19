from agents.llm import invoke_llm
from schemas.state import NegotiationState, CountryState
import json


INPUT_PARSER_PROMPT = """
You are an expert international policy analyst.

The user describes a geopolitical negotiation scenario.
Your job is to extract or infer FOUR negotiating parties from the description.

IMPORTANT NAMING RULES:
- If the user gives explicit names (e.g. "USA", "China"), use those exact names.
- If the user refers to parties as "Country A", "Country B" etc., keep those generic names exactly — do NOT replace them with real country names.
- If fewer than four parties are described, invent plausible FICTIONAL additional parties using generic names like "Country E" or "Regional Bloc D" — never substitute real nation names unless the user provided them.

Return STRICT JSON ONLY — no markdown, no explanation:

{{
  "scenario": "...",
  "countries": {{
    "A": {{
      "name": "...",
      "resources": [],
      "problems": [],
      "goals": []
    }},
    "B": {{
      "name": "...",
      "resources": [],
      "problems": [],
      "goals": []
    }},
    "C": {{
      "name": "...",
      "resources": [],
      "problems": [],
      "goals": []
    }},
    "D": {{
      "name": "...",
      "resources": [],
      "problems": [],
      "goals": []
    }}
  }}
}}

User description:
{text}
"""


def parse_natural_language_input(text: str) -> NegotiationState:
    print("DEBUG: Calling Groq to parse natural language input...")

    response = invoke_llm(INPUT_PARSER_PROMPT.format(text=text), agent_id="A")
    print("DEBUG: Groq returned response")

    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM failed to return valid JSON.\n\nResponse:\n{response}"
        )

    countries = {}
    for key in ("A", "B", "C", "D"):
        c = data["countries"][key]
        countries[key] = CountryState(
            name=c["name"],
            resources=c.get("resources", []),
            problems=c.get("problems", []),
            goals=c.get("goals", []),
        )

    return NegotiationState(
        scenario=data["scenario"],
        countries=countries,
    )