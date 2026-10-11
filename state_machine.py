import asyncio
import operator
import os
import boto3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrockConverse

load_dotenv()

class AppState(TypedDict):
    job_desc: str
    cv_draft: str
    feedback: Annotated[list[str], operator.add]
    loops: int

# Remove the system-level bearer token set by Claude Code. This process is a
# child and removing it here does not affect Claude Code's own process.
# Without this, botocore negotiates bearer auth and ignores our IAM credentials.
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

_bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# Claude 4/5 models require cross-region inference profiles (us. prefix)
llm = ChatBedrockConverse(
    client=_bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-6",
    temperature=0.0,
)

async def scrape_jobs(state: AppState):
    return {"job_desc": "Machine Learning Engineer at OpenAI"}

async def draft_cv(state: AppState):
    loops = state.get("loops", 0) + 1
    prompt = f"You are an aerospace recruiter. Write a CV for this job description: {state.get('job_desc')}"
    response = await llm.ainvoke(prompt)
    return {"cv_draft": response.content, "loops": loops}

async def review_draft(state: AppState):
    loops = state.get("loops", 0)
    prompt = f"Review this CV draft: {state.get('cv_draft')} against the job description: {state.get('job_desc')}.if approved say 'Approved', otherwise provide feedback."
    response = await llm.ainvoke(prompt)
    
    if loops < 2:
        return {"feedback": [response.content]}
    else:
        return {"feedback": ["Approved"]}

def route_next(state: AppState):
    if "approved" in state["feedback"][-1].lower():
        return END
    return "drafter"

workflow = StateGraph(AppState)
workflow.add_node("scraper", scrape_jobs)
workflow.add_node("drafter", draft_cv)
workflow.add_node("reviewer", review_draft)

workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "drafter")
workflow.add_edge("drafter", "reviewer")
workflow.add_conditional_edges("reviewer", route_next, {END: END, "drafter": "drafter"})

app = workflow.compile()

async def main():
    final_state = await app.ainvoke({"loops": 0, "feedback": []})
    print("\n--- Final State ---")
    print(f"Draft: {final_state['cv_draft']}")
    print(f"Feedback History: {final_state['feedback']}")

if __name__ == "__main__":
    asyncio.run(main())