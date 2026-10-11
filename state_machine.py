import asyncio
import operator
import os
import boto3
import numpy as np
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrockConverse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

portfolio = {
    "alrs_007_capsule": "C++ firmware dual-MCU architecture. I2C sensor data logging. IMU integration. Finite state machine for parachute descent control and telemetry payload.",
    "autonomous_terraforming_robot": "C++ kinematic control loops. Arduino GIGA R1 dual-core microcontroller. LiDAR obstacle avoidance and spatial mapping. Hardware PWM actuator control.",
    "aegis_swarm": "ROS 2 Jazzy distributed drone swarm autonomy. FastDDS middleware configuration. NVIDIA Isaac Sim digital twin simulation. Meta Quest 3 OpenXR teleoperation via Unity C#.",
    "comp0211_control_theory": "Mathematical control theory. Linear Quadratic Regulators (LQR), Model Predictive Control (MPC), Sliding Mode Control. Lyapunov stability analysis and Control Barrier Functions in Python.",
    "skimlit_pubmed": "Python TensorFlow deep learning. Natural language processing classification of biomedical RCT abstract sentences. Data preprocessing pipelines.",
    "c_terminal_game": "C programming memory management. Pointers, structs, and terminal I/O game loop.",
    "robotics_challenge": "Electronics hardware integration. Breadboard circuit wiring, microcontroller logic, CAD Autodesk Fusion.",
    "job_drop_ai_agent": "Python LangGraph multi-agent state machine architecture. AWS Bedrock Claude inference integration. Local Retrieval-Augmented Generation (RAG) using sentence-transformers and cosine similarity vector triage. Human-in-the-Loop (HITL) asynchronous execution."
}

projects = list(portfolio.keys())
enriched_descriptions = [f"{k.replace('_', ' ')}: {v}" for k, v in portfolio.items()]
project_embeddings = model.encode(enriched_descriptions)

class AppState(TypedDict):
    job_desc: str
    cv_draft: str
    feedback: Annotated[list[str], operator.add]
    loops: int
    triage_passed: bool
    retrieved_projects: str
    static_bio: str

os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

_bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

llm = ChatBedrockConverse(
    client=_bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-6",
    temperature=0.0,
)

async def scrape_jobs(state: AppState):
    return {"job_desc": "Seeking robotics engineer with ROS 2, simulation experience, and C++ control loops."}

async def triage_job(state: AppState):
    job = state.get("job_desc", "")
    q_emb = model.encode([job])
    
    scores = cosine_similarity(q_emb, project_embeddings)[0]
    
    top_2_idx = np.argsort(scores)[-2:][::-1]
    best_score = scores[top_2_idx[0]]
    
    if best_score >= 0.40:
        retrieved = "\n".join([enriched_descriptions[i] for i in top_2_idx])
        return {"triage_passed": True, "retrieved_projects": retrieved}
    
    return {"triage_passed": False, "retrieved_projects": ""}

def route_triage(state: AppState):
    if state.get("triage_passed"):
        return "drafter"
    return END

async def draft_cv(state: AppState):
    loops = state.get("loops", 0) + 1
    bio = state.get("static_bio", "")
    proj = state.get("retrieved_projects", "")
    job = state.get("job_desc", "")
    
    prompt = f"You are an aerospace recruiter. Write a CV for this job description: {job}. Use this bio: {bio}. Include these specific projects: {proj}. Don't make up fake experience."
    response = await llm.ainvoke(prompt)
    
    return {"cv_draft": response.content, "loops": loops}

async def review_draft(state: AppState):
    loops = state.get("loops", 0)
    draft = state.get("cv_draft", "")
    job = state.get("job_desc", "")
    
    prompt = f"Review this CV draft: {draft} against the job description: {job}. If approved say 'Approved', otherwise provide feedback."
    response = await llm.ainvoke(prompt)
    
    if loops < 2:
        return {"feedback": [response.content]}
    return {"feedback": ["Approved"]}

def route_next(state: AppState):
    if "approved" in state.get("feedback", [""])[-1].lower():
        return END
    return "drafter"

workflow = StateGraph(AppState)
workflow.add_node("scraper", scrape_jobs)
workflow.add_node("triage", triage_job)
workflow.add_node("drafter", draft_cv)
workflow.add_node("reviewer", review_draft)

workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "triage")
workflow.add_conditional_edges("triage", route_triage, {END: END, "drafter": "drafter"})
workflow.add_edge("drafter", "reviewer")
workflow.add_conditional_edges("reviewer", route_next, {END: END, "drafter": "drafter"})

app = workflow.compile()

async def main():
    initial_state = {
        "loops": 0, 
        "feedback": [],
        "static_bio": "UCL MEng Robotics and Artificial Intelligence student, graduating 2029.",
        "job_desc": "",
        "cv_draft": "",
        "triage_passed": False,
        "retrieved_projects": ""
    }
    final_state = await app.ainvoke(initial_state)
    
    print("\n--- Final State ---")
    if not final_state.get("triage_passed"):
        print("Job dropped at triage phase.")
    else:
        print(f"Draft: {final_state['cv_draft']}")

if __name__ == "__main__":
    asyncio.run(main())