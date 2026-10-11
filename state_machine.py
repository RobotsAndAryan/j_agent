from typing import TypedDict
from langgraph.graph import StateGraph

class AppState(TypedDict):
    job_desc: str
    cv_draft : str
    feedback: list[str]
    loops: int

def scrape_jobs(state: AppState):
    return {"job_desc": "Machine Learning Engineer at OpenAI"}}

def draft_cv(state: AppState):
    return {"cv_draft": "Draft CV for Machine Learning Engineer position"}

def review_draft(state: AppState):
   return {"loops": state.get("loops", 0) + 1}