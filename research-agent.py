import json
import sqlite3
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

# ── LangChain community tools ─────────────────────────────────────────────────
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

# ── Model ─────────────────────────────────────────────────────────────────────
from langchain_ollama import ChatOllama

# ── Typing ────────────────────────────────────────────────────────────────────
from typing_extensions import TypedDict


# ══════════════════════════════════════════════════════════════════════════════
# 1. RESEARCH TOOLS
# ══════════════════════════════════════════════════════════════════════════════

ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_web_search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description=(
        "Search the web for current information using DuckDuckGo. "
        "Use for recent news, articles, websites, and real-time data."
    ),
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description=(
        "Query Wikipedia for encyclopedic knowledge on people, places, "
        "events, concepts, and historical topics."
    ),
)

arxiv_wrapper = ArxivAPIWrapper()
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
    description=(
        "Search arXiv for academic research papers on scientific, "
        "technical, ML, physics, math, or CS topics."
    ),
)


@tool
def get_current_datetime() -> str:
    """Return the current date and time for time-sensitive context."""
    return datetime.now().strftime("%A, %B %d, %Y %I:%M %p")


TOOLS = [ddgs_web_search_tool, wiki_tool, arxiv_tool, get_current_datetime]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

SYSTEM_RESEARCH_PROMPT = """You are an expert research assistant with access to \
web search, Wikipedia, arXiv, and a datetime tool. Your goal is to provide \
thorough, accurate, and well-sourced answers.

Research Strategy:
- Use web_search for current events and general information.
- Use wikipedia for established facts and historical background.
- Use arxiv for scientific papers and academic research.
- Use get_current_datetime when time-sensitive context is needed.
- Cross-reference multiple sources for accuracy.

Response Guidelines:
- Cite sources clearly.
- Distinguish facts from uncertain/emerging findings.
- Use structured, easy-to-read formatting.
- Provide a concise summary followed by detailed findings.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════════════════════
# 3. HITL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _display_tool_call(tool_name: str, tool_args: dict) -> None:
    print("\n" + "═" * 60)
    print("  🔍  PENDING TOOL CALL — HUMAN REVIEW REQUIRED")
    print("═" * 60)
    print(f"  Tool   : {tool_name}")
    args_str = json.dumps(tool_args, indent=4)
    for line in args_str.splitlines():
        print(f"  {line}")
    print("═" * 60)


def _get_human_decision(tool_name: str, tool_args: dict) -> dict:
    while True:
        print("\n  Options:")
        print("    [A] Approve  — run the tool as-is")
        print("    [E] Edit     — modify arguments before running")
        print("    [R] Reject   — skip this tool call")
        choice = input("\n  Your choice (A/E/R): ").strip().upper()

        if choice == "A":
            return {"action": "approve", "args": tool_args}

        elif choice == "E":
            print(f"\n  Current args: {json.dumps(tool_args, indent=4)}")
            print("  Enter new args as JSON (press Enter to keep current):")
            raw = input("  > ").strip()
            if not raw:
                return {"action": "approve", "args": tool_args}
            try:
                new_args = json.loads(raw)
                print(f"\n  ✅  Edited args accepted: {new_args}")
                return {"action": "edit", "args": new_args}
            except json.JSONDecodeError:
                print("  ⚠️  Invalid JSON — please try again.")

        elif choice == "R":
            reason = input("  Rejection reason (optional): ").strip()
            return {
                "action": "reject",
                "args": tool_args,
                "reason": reason or "No reason provided.",
            }
        else:
            print("  ⚠️  Invalid choice. Please enter A, E, or R.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState, llm_with_tools) -> dict:
    """LLM reasoning node — produces a final answer OR a tool call."""
    messages = [SystemMessage(content=SYSTEM_RESEARCH_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def hitl_node(state: AgentState) -> Command:
    """HITL gate — pauses graph, collects human decision, resumes."""
    last_msg = state["messages"][-1]

    if not getattr(last_msg, "tool_calls", None):
        return Command(goto="agent")

    tool_call = last_msg.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args  = tool_call["args"]

    _display_tool_call(tool_name, tool_args)
    decision = _get_human_decision(tool_name, tool_args)

    return Command(resume=decision)


def execute_tool_node(state: AgentState) -> dict:
    """Runs or skips the tool; injects ToolMessage back into state."""
    last_msg     = state["messages"][-1]
    tool_call    = last_msg.tool_calls[0]
    tool_name    = tool_call["name"]
    tool_call_id = tool_call["id"]

    decision: dict = interrupt("awaiting_hitl_decision")
    action = decision.get("action")
    args   = decision.get("args", tool_call["args"])

    if action == "reject":
        reason = decision.get("reason", "Rejected by human.")
        print(f"\n  ❌  Tool '{tool_name}' rejected. Reason: {reason}")
        tool_result = (
            f"[Tool '{tool_name}' was rejected by the human operator. "
            f"Reason: {reason}. Please adjust your approach.]"
        )
    else:
        label = "✏️  Edited args —" if action == "edit" else "✅ "
        print(f"\n  {label} Running '{tool_name}'...")
        selected_tool = TOOLS_BY_NAME.get(tool_name)
        if selected_tool is None:
            tool_result = f"[Unknown tool: {tool_name}]"
        else:
            try:
                tool_result = selected_tool.invoke(args)
            except Exception as exc:
                tool_result = f"[Tool error: {exc}]"
        preview = str(tool_result)[:250]
        print(f"\n  📦  Result preview: {preview}{'...' if len(str(tool_result)) > 250 else ''}")

    return {
        "messages": [
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "hitl"
    return END


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(llm_with_tools, db_path: str = "research_agent_pro.db"):
    builder = StateGraph(AgentState)

    builder.add_node("agent",        lambda state: agent_node(state, llm_with_tools))
    builder.add_node("hitl",         hitl_node)
    builder.add_node("execute_tool", execute_tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"hitl": "hitl", END: END},
    )
    builder.add_edge("hitl",         "execute_tool")
    builder.add_edge("execute_tool", "agent")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],
    )
    return graph


# ══════════════════════════════════════════════════════════════════════════════
# 7. STREAMING — with proper HITL interrupt-resume loop
# ══════════════════════════════════════════════════════════════════════════════

def _stream_until_interrupt(graph, inputs, config: dict) -> bool:
    """
    Stream graph until it finishes OR hits an interrupt (HITL gate).
    Returns True  → graph paused at interrupt, human decision needed.
    Returns False → graph finished, final answer delivered.
    """
    for chunk in graph.stream(inputs, config=config, stream_mode="values"):
        last = chunk["messages"][-1]
        if isinstance(last, AIMessage):
            if last.content:
                print(f"\n🤖  Agent: {last.content}")
            elif last.tool_calls:
                names = [tc["name"] for tc in last.tool_calls]
                print(f"\n⚙️   Selecting tool(s): {names}")

    # state.next is non-empty only when graph is interrupted mid-run
    state = graph.get_state(config)
    return bool(state.next)


def run_query(graph, query: str, config: dict) -> None:
    """
    Full HITL execution loop for one user query:
      1. Send query  → stream → may interrupt at HITL gate
      2. Show tool call → human approves / edits / rejects
      3. Resume graph with Command(resume=decision) → stream again
      4. Repeat until no more interrupts → final answer shown
    """
    print()

    # ── Step 1: initial run ───────────────────────────────────────────────────
    interrupted = _stream_until_interrupt(
        graph,
        {"messages": [HumanMessage(content=query)]},
        config,
    )

    # ── Steps 2-4: HITL approval loop ────────────────────────────────────────
    while interrupted:
        state    = graph.get_state(config)
        last_msg = state.values["messages"][-1]

        # Read the pending tool call from the checkpointed state
        if not getattr(last_msg, "tool_calls", None):
            break

        tool_call = last_msg.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Show HITL prompt — human decides
        _display_tool_call(tool_name, tool_args)
        decision = _get_human_decision(tool_name, tool_args)

        # Resume graph from checkpoint with decision
        interrupted = _stream_until_interrupt(
            graph,
            Command(resume=decision),
            config,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8. BANNER + MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          RESEARCH AGENT PRO  —  HITL Edition             ║
║   Human-in-the-Loop  ·  SqliteSaver  ·  LangGraph        ║
╠══════════════════════════════════════════════════════════╣
║  Model  : gpt-oss:20b  (Ollama)                          ║
║  Tools  : DuckDuckGo · Wikipedia · arXiv · DateTime      ║
║  Control: [A]pprove / [E]dit / [R]eject every tool call  ║
║  State  : persisted to research_agent_pro.db             ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    graph  = build_graph(llm_with_tools)
    config = {"configurable": {"thread_id": "research-pro-session-1"}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy researching! 🔬")
            break

        try:
            run_query(graph, query, config)
        except Exception as err:
            print(f"\n⚠️  Error: {err}")


if __name__ == "__main__":
    main()
