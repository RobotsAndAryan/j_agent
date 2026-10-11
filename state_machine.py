import asyncio
import operator
import os
import boto3
import numpy as np
import markdown
import pdfkit
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_aws import ChatBedrockConverse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

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
    approved_by_human: bool
    pdf_path: str  # NEW: Tracks the location of the compiled file

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
    
    prompt = f"You are an aerospace recruiter who is the best in the industry. Write a CV for this job description: {job}. Use this bio: {bio}. Include these specific projects: {proj}. Don't make up fake experience and use google's XYZ Formula."
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
    # CHANGED: Now routes to compile_pdf instead of submit directly
    if "approved" in state.get("feedback", [""])[-1].lower():
        return "compile_pdf" 
    return "drafter"

async def compile_pdf(state: AppState):
    # NEW: The PDF Compiler Node
    if not state.get("approved_by_human"):
        return {"pdf_path": ""}
        
    print("\n>>> COMPILING CV TO PDF... <<<")
    md_text = state.get("cv_draft", "")
    
    # 1. ATS-Compliant CSS styling
    css = """
    <style>
        body { font-family: 'Arial', sans-serif; font-size: 11pt; color: #000; line-height: 1.4; margin: 0.8in; }
        h1 { font-size: 18pt; text-align: center; border-bottom: 2px solid #000; margin-bottom: 10px; padding-bottom: 5px; }
        h2 { font-size: 13pt; border-bottom: 1px solid #999; margin-top: 20px; margin-bottom: 10px; padding-bottom: 3px;}
        h3 { font-size: 11pt; margin-bottom: 2px; }
        p, ul, li { margin-top: 2px; margin-bottom: 5px; }
        hr { display: none; }
    </style>
    """
    
    # 2. Markdown to HTML
    html_body = markdown.markdown(md_text)
    full_html = f"<html><head><meta charset='utf-8'>{css}</head><body>{html_body}</body></html>"
    
    # 3. HTML to PDF
    output_filename = "orbital_drop_cv.pdf"
    options = {
        'page-size': 'A4',
        'margin-top': '0',
        'margin-right': '0',
        'margin-bottom': '0',
        'margin-left': '0',
        'encoding': "UTF-8",
        'disable-smart-shrinking': ''
    }
    
    try:
        pdfkit.from_string(full_html, output_filename, options=options)
        print(f">>> PDF SUCCESSFULLY RENDERED: {os.path.abspath(output_filename)} <<<")
    except Exception as e:
        print(f">>> PDF COMPILATION FAILED: {e} <<<")
        
    return {"pdf_path": output_filename}

async def submit_application(state: AppState):
    if state.get("approved_by_human"):
        print(f"\n>>> AUTONOMOUS SUBMISSION EXECUTING WITH PAYLOAD: {state.get('pdf_path')} <<<")
    else:
        print("\n>>> SUBMISSION ABORTED. NO HUMAN APPROVAL. <<<")
    return {}

workflow = StateGraph(AppState)
workflow.add_node("scraper", scrape_jobs)
workflow.add_node("triage", triage_job)
workflow.add_node("drafter", draft_cv)
workflow.add_node("reviewer", review_draft)
workflow.add_node("compile_pdf", compile_pdf) # NEW
workflow.add_node("submit_application", submit_application)

workflow.set_entry_point("scraper")
workflow.add_edge("scraper", "triage")
workflow.add_conditional_edges("triage", route_triage, {END: END, "drafter": "drafter"})
workflow.add_edge("drafter", "reviewer")
# CHANGED: Graph flows to compiler upon approval
workflow.add_conditional_edges("reviewer", route_next, {"compile_pdf": "compile_pdf", "drafter": "drafter"})
workflow.add_edge("compile_pdf", "submit_application")
workflow.add_edge("submit_application", END)

memory = MemorySaver()
# CHANGED: The interrupt now sits right before compilation
app = workflow.compile(checkpointer=memory, interrupt_before=["compile_pdf"])

def send_telegram_alert(cv_text):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping Telegram alert.")
        return

    preview = cv_text[:600].replace("`", "'")
    message = (
        "🤖 *JOB DROP AGENT — CV Ready for Review*\n\n"
        f"```\n{preview}\n```\n\n"
        "_Reply YES in the terminal to compile PDF and submit._"
    )

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"Telegram alert sent (message_id: {resp.json()['result']['message_id']})")
    except Exception as e:
        print(f"Telegram alert failed ({e}). Review CV above and approve via terminal.")

async def main():
    initial_state = {
        "loops": 0, 
        "feedback": [],
        "static_bio": "Name: Aryan Naik | Email: aryannaik1205@gmail.com | GitHub: github.com/RobotsAndAryan | Education: 2nd-year MEng Robotics and Artificial Intelligence at University College London (UCL) | Graduation: 2027.",
        "job_desc": "",
        "cv_draft": "",
        "triage_passed": False,
        "retrieved_projects": "",
        "approved_by_human": False,
        "pdf_path": ""
    }
    
    thread_config = {"configurable": {"thread_id": "job_application_thread_001"}}
    
    print("Starting graph...")
    async for event in app.astream(initial_state, thread_config, stream_mode="values"):
        pass

    state_snapshot = app.get_state(thread_config)

    if not state_snapshot.values.get("triage_passed"):
        print("Job dropped at triage phase.")
        return

    # CHANGED: We now intercept at the compile_pdf node
    if state_snapshot.next == ('compile_pdf',):
        print("\n--- GRAPH PAUSED FOR HITL ---")
        current_cv = state_snapshot.values.get("cv_draft", "")
        print(f"\n--- CV DRAFT ---\n{current_cv}\n--- END CV ---")
        send_telegram_alert(current_cv)

        user_input = input("\nTelegram alert sent. Do you approve this CV for compilation and submission? (yes/no): ")
        
        if user_input.strip().lower() == "yes":
            app.update_state(thread_config, {"approved_by_human": True})
            print("Resuming graph...")
            async for event in app.astream(None, thread_config, stream_mode="values"):
                pass
        else:
            print("Graph terminated by user.")

if __name__ == "__main__":
    asyncio.run(main())