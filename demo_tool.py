import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

console = Console()

WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = WORKSPACE / "local_agent_report.md"
MLX_CHAT_URL = os.getenv("MLX_CHAT_URL", "http://localhost:8080/v1/chat/completions")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "").strip()


class AgentState(TypedDict):
    task: str
    city: str
    search_query: str
    route: str
    weather_result: str
    search_result: str
    final_answer: str
    output_file: str


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def call_local_model(system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    if LOCAL_MODEL_NAME:
        payload["model"] = LOCAL_MODEL_NAME

    response = requests.post(MLX_CHAT_URL, json=payload, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(f"MLX server error: {response.status_code}\n{response.text}")

    data = response.json()
    message = data["choices"][0]["message"]

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)

    return strip_thinking(content)


def route_task(state: AgentState) -> AgentState:
    console.rule("Step 1 - Tool Routing")

    city = state["city"].strip()
    search_query = state["search_query"].strip()

    if city and search_query:
        route = "WEATHER_AND_SEARCH"
    elif city:
        route = "WEATHER_ONLY"
    elif search_query:
        route = "SEARCH_ONLY"
    else:
        route = "LLM_ONLY"

    console.print(Panel(f"Route selected: {route}", title="Router"))

    return {**state, "route": route}


def weather_tool(state: AgentState) -> AgentState:
    console.rule("Step 2 - Weather Tool")

    city = state["city"].strip()

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=30,
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()

    if not geo_data.get("results"):
        result = f"No location found for city: {city}"
        console.print(Panel(result, title="Weather Result"))
        return {**state, "weather_result": result}

    location = geo_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability,rain",
            "forecast_days": 2,
            "timezone": "auto",
        },
        timeout=30,
    )
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    hourly = weather_data.get("hourly", {})
    next_12_hours = []

    times = hourly.get("time", [])[:12]
    temps = hourly.get("temperature_2m", [])[:12]
    rain_probs = hourly.get("precipitation_probability", [])[:12]
    rains = hourly.get("rain", [])[:12]

    for i, time_value in enumerate(times):
        next_12_hours.append(
            {
                "time": time_value,
                "temperature_c": temps[i] if i < len(temps) else None,
                "rain_probability_percent": rain_probs[i] if i < len(rain_probs) else None,
                "rain_mm": rains[i] if i < len(rains) else None,
            }
        )

    result = {
        "location": {
            "name": location.get("name", city),
            "region": location.get("admin1", ""),
            "country": location.get("country", ""),
            "latitude": latitude,
            "longitude": longitude,
        },
        "current": weather_data.get("current", {}),
        "next_12_hours": next_12_hours,
    }

    result_text = json.dumps(result, indent=2, ensure_ascii=False)
    console.print(Panel(result_text, title="Weather API Result"))

    return {**state, "weather_result": result_text}


def search_tool(state: AgentState) -> AgentState:
    console.rule("Step 3 - Search Tool")

    query = state["search_query"].strip()

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        },
        timeout=30,
        headers={"User-Agent": "LocalMLXAgentDemo/1.0"},
    )
    response.raise_for_status()

    data = response.json()
    titles = data[1] if len(data) > 1 else []
    snippets = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []

    results = []
    for title, snippet, url in zip(titles, snippets, urls):
        clean_snippet = re.sub("<.*?>", "", snippet)
        clean_snippet = html.unescape(clean_snippet)
        results.append({"title": title, "snippet": clean_snippet, "url": url})

    result_text = json.dumps(results, indent=2, ensure_ascii=False)
    console.print(Panel(result_text, title="Search API Result"))

    return {**state, "search_result": result_text}


def synthesize_answer(state: AgentState) -> AgentState:
    console.rule("Step 4 - Local MLX Model Synthesis")

    system_prompt = """
You are a local AI agent running on a Mac using MLX.
Use only the tool results given to you.
Give a practical answer for the user.
Mention which tools were used.
Do not output JSON.
Do not write code.
"""

    user_prompt = """
User task:
{task}

Route selected:
{route}

Weather tool result:
{weather_result}

Search tool result:
{search_result}

Create the final answer with these sections:
1. Quick Recommendation
2. Weather-Based Advice
3. Search-Based Context
4. Practical Next Steps
5. Tool Trace
""".format(
        task=state["task"],
        route=state["route"],
        weather_result=state["weather_result"] or "No weather result.",
        search_result=state["search_result"] or "No search result.",
    )

    try:
        answer = call_local_model(system_prompt, user_prompt, max_tokens=1000)
    except Exception as error:
        answer = (
            "The external tools ran, but local model synthesis failed.\n\n"
            f"Error: {type(error).__name__}: {error}\n\n"
            f"Weather result:\n{state['weather_result']}\n\n"
            f"Search result:\n{state['search_result']}"
        )

    console.print(Panel(answer, title="Final Local Model Answer"))
    return {**state, "final_answer": answer}


def save_report(state: AgentState) -> AgentState:
    console.rule("Step 5 - Save Report")

    report_lines = [
        "# Local Agent Report",
        "",
        "## User Task",
        state.get("task", ""),
        "",
        "## Route Selected",
        state.get("route", ""),
        "",
        "## Weather Tool Result",
        state.get("weather_result", "") or "Not used",
        "",
        "## Search Tool Result",
        state.get("search_result", "") or "Not used",
        "",
        "## Final Answer",
        state.get("final_answer", ""),
        "",
    ]

    report = "\n".join(report_lines)
    OUTPUT_FILE.write_text(report, encoding="utf-8")

    console.print(Panel(str(OUTPUT_FILE), title="Saved Report"))
    return {**state, "output_file": str(OUTPUT_FILE)}


def choose_after_route(state: AgentState) -> str:
    if state["route"] in ("WEATHER_AND_SEARCH", "WEATHER_ONLY"):
        return "weather_tool"
    if state["route"] == "SEARCH_ONLY":
        return "search_tool"
    return "synthesize_answer"


def choose_after_weather(state: AgentState) -> str:
    if state["route"] == "WEATHER_AND_SEARCH":
        return "search_tool"
    return "synthesize_answer"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route_task", route_task)
    graph.add_node("weather_tool", weather_tool)
    graph.add_node("search_tool", search_tool)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("save_report", save_report)

    graph.add_edge(START, "route_task")

    graph.add_conditional_edges(
        "route_task",
        choose_after_route,
        {
            "weather_tool": "weather_tool",
            "search_tool": "search_tool",
            "synthesize_answer": "synthesize_answer",
        },
    )

    graph.add_conditional_edges(
        "weather_tool",
        choose_after_weather,
        {
            "search_tool": "search_tool",
            "synthesize_answer": "synthesize_answer",
        },
    )

    graph.add_edge("search_tool", "synthesize_answer")
    graph.add_edge("synthesize_answer", "save_report")
    graph.add_edge("save_report", END)

    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Local MLX + LangGraph tool routing demo")
    parser.add_argument("--task", required=True)
    parser.add_argument("--city", default="")
    parser.add_argument("--search", default="")
    args = parser.parse_args()

    initial_state: AgentState = {
        "task": args.task,
        "city": args.city,
        "search_query": args.search,
        "route": "",
        "weather_result": "",
        "search_result": "",
        "final_answer": "",
        "output_file": "",
    }

    console.print(Panel(args.task, title="User Task"))

    app = build_graph()
    final_state = app.invoke(initial_state)

    console.rule("Done")
    console.print(Panel(final_state["final_answer"], title="Final Answer"))
    console.print(f"Saved report: {final_state['output_file']}")


if __name__ == "__main__":
    main()
