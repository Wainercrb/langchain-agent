import logging
import os
from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate API key at startup
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to .env")

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1024
)


@tool
def get_weather(location: str) -> str:
    """
    Fetch current weather information for a given location.
    
    Args:
        location: The city or location name to get weather for
        
    Returns:
        A string describing the current weather conditions
    """
    # Placeholder implementation - in production, call a real weather API
    logger.info(f"Fetching weather for {location}")
    return f"The current weather in {location} is sunny with a temperature of 25°C."


def create_agent_graph():
    """Create and return a configured agent graph."""
    
    system_prompt = """You are a helpful weather assistant. 
    You provide accurate, concise weather information based on user queries.
    Always be friendly and helpful."""
    
    # Create the agent using LangChain 1.3.1 pattern
    agent = create_agent(
        model=gemini_model,
        tools=[get_weather],
        system_prompt=system_prompt,
        debug=False
    )
    
    return agent


def main() -> None:
    """Main entry point - run the weather agent."""
    try:
        logger.info("Initializing weather agent...")
        agent = create_agent_graph()
        
        # User query
        user_query = "What's the weather like in New York?"
        logger.info(f"User query: {user_query}")
        
        # Invoke agent with input
        response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
        
        # Extract and display result
        logger.info(f"Agent response: {response}")
        print(f"\n✅ Agent Response:\n{response}")
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()