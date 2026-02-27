import os

from async_timeout import timeout
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.6,
    timeout=45
)

def invoke_llm(prompt: str) -> str:
    """
    Sends a prompt to the Groq LLM and returns the text response.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()