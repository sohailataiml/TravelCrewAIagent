"""
Travel Planner Agent using CrewAI + Anthropic.

Multi-agent crew that creates personalized travel itineraries:
- Destination Researcher: gathers destination info
- Activity Planner: creates day-by-day activities
- Budget Analyst: estimates costs

Usage:
    python agent.py --destination "Tokyo, Japan" --days 5 --budget 2000
"""

import argparse
from dotenv import load_dotenv

# Prefer CrewAI when available; otherwise fall back to direct Anthropic usage
try:
    from crewai import Agent, Crew, Process, Task  # type: ignore
    HAVE_CREWAI = True
except Exception:
    HAVE_CREWAI = False

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def call_agent(llm, role: str, goal: str, backstory: str, task_description: str, context_info: str = "") -> str:
    system_prompt = f"""You are a {role}.
Your goal: {goal}
Your background: {backstory}

{f"Context from previous research: {context_info}" if context_info else ""}"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=task_description)]
    resp = llm.invoke(messages)
    return resp.content


def build_travel_crew(destination: str, days: int, budget: float, interests: str) -> str:
    # Use CrewAI orchestration when available; otherwise fallback to sequential calls.
    llm = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0.4)

    if HAVE_CREWAI:
        researcher = Agent(
            role="Destination Researcher",
            goal=f"Research {destination} and provide key travel insights",
            backstory="Expert travel journalist who has visited 100+ countries. Knows the best hidden gems and practical tips.",
            llm=llm,
            verbose=False,
        )

        planner = Agent(
            role="Travel Itinerary Planner",
            goal=f"Create a detailed {days}-day itinerary for {destination}",
            backstory="Luxury travel consultant with 15 years of experience crafting personalized itineraries.",
            llm=llm,
            verbose=False,
        )

        budget_analyst = Agent(
            role="Travel Budget Analyst",
            goal=f"Estimate realistic costs for the trip within ${budget} budget",
            backstory="Financial travel advisor who helps travelers maximize experiences within budget.",
            llm=llm,
            verbose=False,
        )

        research_task = Task(
            description=f"""Research {destination} for a {days}-day trip.
Cover: best time to visit, neighborhoods to stay in, must-see attractions,
local food scene, transportation tips, and cultural customs to know.
Traveler interests: {interests}""",
            agent=researcher,
            expected_output="Destination brief with key areas, attractions, food, and practical tips",
        )

        planning_task = Task(
            description=f"""Create a {days}-day itinerary for {destination}.
Budget: ${budget} total. Interests: {interests}.
Include morning/afternoon/evening activities, specific restaurant recommendations,
and travel time between locations. Make it achievable and enjoyable.""",
            agent=planner,
            expected_output=f"Day-by-day {days}-day itinerary with activities, meals, and timing",
            context=[research_task],
        )

        budget_task = Task(
            description=f"""Provide a budget breakdown for the {days}-day {destination} trip.
Total budget: ${budget}. Include: flights (estimate), accommodation, food, activities,
transportation. Flag if budget is tight and suggest adjustments.""",
            agent=budget_analyst,
            expected_output="Itemized budget breakdown with daily averages and money-saving tips",
            context=[research_task, planning_task],
        )

        crew = Crew(
            agents=[researcher, planner, budget_analyst],
            tasks=[research_task, planning_task, budget_task],
            process=Process.sequential,
            verbose=False,
        )

        return str(crew.kickoff())

    # Fallback: sequential direct LLM calls (works without CrewAI)
    research_output = call_agent(
        llm,
        role="Destination Researcher",
        goal=f"Research {destination} and provide key travel insights",
        backstory="Expert travel journalist who has visited 100+ countries. Knows the best hidden gems and practical tips.",
        task_description=f"""Research {destination} for a {days}-day trip.
Cover: best time to visit, neighborhoods to stay in, must-see attractions,
local food scene, transportation tips, and cultural customs to know.
Traveler interests: {interests}""",
    )

    itinerary_output = call_agent(
        llm,
        role="Travel Itinerary Planner",
        goal=f"Create a detailed {days}-day itinerary for {destination}",
        backstory="Luxury travel consultant with 15 years of experience crafting personalized itineraries.",
        task_description=f"""Create a {days}-day itinerary for {destination}.
Budget: ${budget} total. Interests: {interests}.
Include morning/afternoon/evening activities, specific restaurant recommendations,
and travel time between locations. Make it achievable and enjoyable.""",
        context_info=research_output,
    )

    budget_output = call_agent(
        llm,
        role="Travel Budget Analyst",
        goal=f"Estimate realistic costs for the trip within ${budget} budget",
        backstory="Financial travel advisor who helps travelers maximize experiences within budget.",
        task_description=f"""Provide a budget breakdown for the {days}-day {destination} trip.
Total budget: ${budget}. Include: flights (estimate), accommodation, food, activities,
transportation. Flag if budget is tight and suggest adjustments.""",
        context_info=f"Destination Info:\n{research_output}\n\nProposed Itinerary:\n{itinerary_output}",
    )

    return f"""
## DESTINATION RESEARCH
{research_output}

## TRAVEL ITINERARY
{itinerary_output}

## BUDGET ANALYSIS
{budget_output}
"""


def main():
    parser = argparse.ArgumentParser(description="Travel Planner Agent")
    parser.add_argument("--destination", required=True, help="Travel destination (e.g., 'Tokyo, Japan')")
    parser.add_argument("--days", type=int, default=7, help="Number of days for the trip (default: 7)")
    parser.add_argument("--budget", type=float, default=3000, help="Budget in USD (default: 3000)")
    parser.add_argument("--interests", default="food, culture, history", help="Trip interests (default: food, culture, history)")

    args = parser.parse_args()

    print(f"\n🌍 Planning your {args.days}-day trip to {args.destination}")
    print(f"   Budget: ${args.budget:,.2f}")
    print(f"   Interests: {args.interests}")
    print("   Researching with Claude Opus...\n")

    result = build_travel_crew(args.destination, args.days, args.budget, args.interests)
    print(result)


if __name__ == "__main__":
    main()