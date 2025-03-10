from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from typing import Iterator
from phi.agent import RunResponse
from phi.utils.pprint import pprint_run_response
import re

def preprocess_text(text: str) -> str:
    """
    Cleans the input text by removing markdown, links, mentions, and other unwanted characters.
    """
    text = re.sub(r'\*[^*]*\*', '', text)  # Remove bold
    text = re.sub(r'\_[^\_]*\_', '', text)  # Remove italics
    text = re.sub(r'`[^`]*`', '', text)  # Remove code blocks
    text = re.sub(r'\[[^]]*\]\([^)]*\)', '', text)  # Remove links
    text = re.sub(r'\#[a-zA-Z0-9-_]+', '', text)  # Remove hashtags
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML/XML tags
    text = re.sub(r'#+', '', text)  # Remove continuous '#' characters
    text = " ".join(text.split())
    return text

query_classifier_agent = Agent(
    name="Query Classifier Agent",
    description="Classifies user queries for appropriate agent routing.",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions="Determine the most suitable agent for the given query.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

def classify_query(user_query: str) -> str:
    """
    Determines the most suitable agent for the given query.
    """
    response_stream: Iterator[RunResponse] = query_classifier_agent.run(
        f"Classify the query '{user_query}' into Web Agent, Code Review Agent, or Quantization Agent.",
        stream=True,
        messages=True
    )
    extracted_text = "".join([message.content for response in response_stream for message in response.messages if isinstance(message.content, str)])
    return preprocess_text(extracted_text)

def route_query(user_query: str) -> str:
    """
    Routes the query and provides a classification result.
    """
    classification = classify_query(user_query)
    print(f"Classified as: {classification}")
    return classification

def run_query_selector():
    """
    Interactive mode to get user query and classify it.
    """
    load_dotenv()
    user_query = input("Enter your query: ")
    return route_query(user_query)

def batch_classify_queries(queries: list) -> dict:
    """
    Classifies multiple queries at once.
    """
    return {query: classify_query(query) for query in queries}

if __name__ == "__main__":
    run_query_selector()
