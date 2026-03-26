# Research AI Agent 
import sqlite3
from datetime import datetime

# LangChain Agent & Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool

# LangChain Community 
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
) 
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# LangGraph
from langgraph.checkpoint.memory import MemorySaver

# Model
from langchain_ollama import ChatOllama


# --- RESEARCH TOOLS ---

ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_web_search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the web for current information using DuckDuckGo. Use this tool to find recent news, articles, websites, and real-time data on any topic. Returns a list of search results with titles, snippets, and URLs."
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Query Wikipedia for encyclopedic knowledge on people, places, events, concepts, and historical topics. Use this tool when you need detailed, factual background information or well-established knowledge. Returns article summaries from Wikipedia."
)

arxiv_wrapper = ArxivAPIWrapper()
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
    description="Search arXiv for academic and scientific research papers. Use this tool when the query involves scientific research, technical papers, machine learning, physics, mathematics, computer science, or other academic disciplines. Returns paper titles, authors, abstracts, and publication details."
)

@tool
def get_current_datetime() -> str:
    """Return the current date and time. Use this tool when you need to know today's date or the current time to provide timely and contextually accurate responses."""
    now = datetime.now()
    return now.strftime(
        "%A, %B %d, %Y %I:%M %p"
    )

tools = [ddgs_web_search_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = """You are an expert research assistant with access to web search, Wikipedia, arXiv, and a datetime tool. Your goal is to provide thorough, accurate, and well-sourced answers to user queries.

## Research Strategy
- Break complex questions into sub-queries and investigate each one.
- Use **web_search** for current events, recent developments, and general information.
- Use **wikipedia** for established facts, historical context, and background knowledge.
- Use **arxiv** for scientific papers, technical research, and academic findings.
- Use **get_current_datetime** when time-sensitive context is needed.
- Cross-reference multiple sources to verify claims and ensure accuracy.

## Response Guidelines
- Cite your sources clearly, referencing where the information came from.
- Distinguish between well-established facts and emerging or uncertain findings.
- Present information in a structured, easy-to-read format.
- If the available information is insufficient or conflicting, state that transparently.
- Provide a concise summary followed by detailed findings when appropriate.
"""

def stream_response(agent, query, config):
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, 
                              config=config,
                              stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                # print(f"User: {latest_message.content}")
                pass
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


# --- MIDDLEWARE --- 

@wrap_tool_call
def tool_handle_error(request, handler):
    try:
        return handler(request) 
    except Exception as err:
        print(f"Error: {err}")


tool_retry = ToolRetryMiddleware(
    max_retries=2,
    tools=["web_search", "arxiv"],
    max_delay=30,
    backoff_factor=2,
    initial_delay=1,
    on_failure="continue"
)

summ_midd = SummarizationMiddleware(
    model="ollama:minimax-m2.5:cloud",
    trigger=("tokens", 2000),
    keep=("messages", 5),
)

model_retry = ModelRetryMiddleware(
    max_retries=2
)

# --- AGENT SETUP --- 

def create_research_agent():

    llm = ChatOllama(
        model="minimax-m2.5:cloud",
        temperature=0
    )

    memory = MemorySaver()

    middleware = [tool_handle_error, tool_retry, summ_midd, model_retry]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        checkpointer=memory,
        middleware=middleware,
        name="research_agent"
    )

    return agent

def banner():
    """"""
    print("Research AI Agent")


def main():
    banner()
    agent = create_research_agent()

    config = {"configurable": {"thread_id": "research-session-1"}}

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy researching!")
            break
        
        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


main()