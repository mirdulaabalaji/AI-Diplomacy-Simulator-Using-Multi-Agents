from agents.llm import invoke_llm
from schemas.state import NegotiationState, CountryState
import json


INPUT_PARSER_PROMPT = """
You are an expert international policy analyst.

The user may describe countries, goals, problems, and resources
in natural language. The scenario may be implicit.

If no explicit scenario is stated, infer a reasonable one.

Return STRICT JSON ONLY in this EXACT format:

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
    }}
  }}
}}

User description:
{text}
"""


def parse_natural_language_input(text: str) -> NegotiationState:
    print("DEBUG: Calling Groq to parse natural language input...")

    response = invoke_llm(
        INPUT_PARSER_PROMPT.format(text=text)
    )
    print("DEBUG: Groq returned response")

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError(
            f"LLM failed to return valid JSON.\n\nResponse:\n{response}"
        )

    countries = {}

    for key, c in data["countries"].items():
        countries[key] = CountryState(
            name=c["name"],
            resources=c["resources"],
            problems=c["problems"],
            goals=c["goals"],
        )

    return NegotiationState(
        scenario=data["scenario"],
        countries=countries
    )