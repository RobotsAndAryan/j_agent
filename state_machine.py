import asyncio
import json
import operator
import os
import re
import sqlite3
import time
import warnings
from datetime import datetime
from urllib.parse import urlparse

import boto3
import markdown
import numpy as np
import pdfkit
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jobspy import scrape_jobs as _jobspy_scrape
from langgraph.graph import END, StateGraph
from langchain_aws import ChatBedrockConverse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Annotated, TypedDict

warnings.filterwarnings("ignore")  # suppress jobspy SSL / pandas FutureWarnings

load_dotenv()

# ── Embedding model ───────────────────────────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2")

portfolio = {
    "alrs_007_capsule": (
        "Aerospace avionics firmware for a student rocketry telemetry payload. "
        "C++ firmware on a dual-MCU embedded architecture with interrupt-driven real-time execution. "
        "I2C serial bus sensor integration for IMU inertial measurement unit data acquisition, "
        "attitude estimation, altitude sensing and acceleration logging. "
        "Finite state machine FSM for autonomous parachute deployment and descent control "
        "under safety-critical flight conditions. "
        "Wireless telemetry data transmission and onboard flash data logging. "
        "Embedded systems, real-time firmware, hardware-software co-design, sensor fusion, "
        "UART SPI serial communication, power budgeting, signal conditioning. "
        "Aerospace engineering, avionics, rocketry, launch vehicle, flight computer, "
        "guidance navigation and control GNC, autonomous recovery systems, "
        "safety-critical embedded software, defence electronics."
    ),
    "autonomous_terraforming_robot": (
        "Autonomous ground robot for terrain mapping and unstructured environment navigation. "
        "C++ real-time kinematic control loops on an Arduino GIGA R1 dual-core microcontroller. "
        "LiDAR-based simultaneous localisation and mapping SLAM for obstacle detection and avoidance. "
        "Hardware PWM motor control for DC drive actuation and servo positioning. "
        "PID feedback control, encoder odometry, path planning, real-time embedded control "
        "on resource-constrained hardware. "
        "Mechatronics, UGV unmanned ground vehicle, autonomous mobile robotics, "
        "motor driver integration, sensor-actuator coupling, mechanical-electrical integration, "
        "autonomous navigation, field robotics, industrial automation concepts, "
        "robotic assembly, manufacturing automation, automotive embedded systems."
    ),
    "aegis_swarm": (
        "Distributed multi-UAV drone swarm autonomy system. "
        "ROS 2 Jazzy robot operating system with C++ nodes for multi-agent coordination "
        "and autonomous aerial navigation. "
        "FastDDS real-time DDS middleware for high-frequency inter-agent pub-sub communication. "
        "NVIDIA Isaac Sim physics-based digital twin for software-in-the-loop SITL "
        "and hardware-in-the-loop HITL simulation and verification. "
        "Meta Quest 3 OpenXR immersive teleoperation interface developed in Unity C# "
        "for 6-DOF operator control and situational awareness. "
        "Unmanned aerial vehicle UAV swarm robotics, defence aerospace, autonomous flight, "
        "multi-agent systems, distributed computing, mission planning, fleet management, "
        "ROS2 middleware, robot simulation, digital twin, extended reality XR, "
        "spatial computing, C++ software engineering."
    ),
    "comp0211_control_theory": (
        "Mathematical control theory coursework implemented in Python. "
        "Linear Quadratic Regulator LQR optimal state feedback controller design "
        "for energy-efficient trajectory tracking. "
        "Model Predictive Control MPC for constrained optimisation and receding horizon planning. "
        "Sliding Mode Control SMC for robust nonlinear system stabilisation. "
        "Lyapunov stability analysis and Control Barrier Functions CBF for safety-critical constraints. "
        "State space modelling, controllability and observability analysis, Kalman filter estimation, "
        "system identification, pole placement, frequency domain Bode plot, root locus. "
        "Applied to quadrotor UAV dynamics, robotic arm trajectory planning, "
        "autonomous vehicle path tracking, aerospace vehicle dynamics. "
        "Optimal control, robust control, nonlinear control, dynamical systems, "
        "MATLAB-equivalent simulation, autonomous systems."
    ),
    "skimlit_pubmed": (
        "Deep learning NLP project in Python using TensorFlow and Keras. "
        "Multi-input neural network for sequential sentence classification of biomedical text, "
        "combining character-level CNN embeddings, token-level word embeddings "
        "and positional encodings for context-aware classification. "
        "End-to-end data preprocessing pipeline: tokenisation, annotation, batching, augmentation. "
        "Model training, validation, hyperparameter tuning, precision-recall evaluation. "
        "Machine learning, artificial intelligence, deep learning, neural networks, NLP, "
        "natural language processing, text classification, transformer concepts, "
        "Python data science, scikit-learn, numpy, pandas, data pipelines, "
        "computer vision adjacent techniques, model evaluation, feature engineering."
    ),
    "c_terminal_game": (
        "Systems-level C programming project: a fully functional terminal game engine. "
        "Manual dynamic memory allocation and deallocation with malloc free, "
        "zero memory leaks verified with Valgrind. "
        "Complex pointer arithmetic, nested struct-based data modelling, "
        "linked list and stack data structures. "
        "Real-time game loop with non-blocking terminal I/O, keyboard event handling, "
        "buffered screen rendering and frame-rate control. "
        "Algorithm design for game state management, collision detection, pathfinding. "
        "Low-level C programming, POSIX systems programming, embedded-capable firmware C, "
        "data structures and algorithms, software engineering principles, "
        "Linux development environment, debugging, memory safety."
    ),
    "robotics_challenge": (
        "Full-system hardware robotics integration project under competition constraints. "
        "Breadboard electronics design integrating microcontrollers, sensors, motor drivers, "
        "actuators and power electronics into a complete functional robotic platform. "
        "Circuit wiring from schematic to physical assembly; systematic fault-finding "
        "and hardware debugging to verify signal integrity. "
        "Microcontroller firmware for closed-loop sensor-actuator feedback control. "
        "CAD mechanical design in Autodesk Fusion 360: structural components, "
        "enclosures, mounting geometry, tolerancing. "
        "PCB layout concepts, component selection, voltage regulation, signal conditioning, "
        "servo and H-bridge motor control, ultrasonic IR distance sensors, "
        "rapid prototyping, 3D printing, systems integration and acceptance testing, "
        "mechatronics, electronics engineering."
    ),
    "job_drop_ai_agent": (
        "Production autonomous multi-agent AI pipeline for automated job application generation. "
        "LangGraph stateful graph orchestration across nodes: scraping, RAG triage, "
        "LLM CV generation, LLM cover letter generation, PDF compilation, SQLite logging. "
        "AWS Bedrock Claude Sonnet 4.6 LLM inference via boto3 and LangChain AWS. "
        "Sentence-transformers all-MiniLM-L6-v2 semantic embedding, "
        "cosine similarity vector search for RAG retrieval-augmented generation triage. "
        "BeautifulSoup4 multi-source web scraping: Reed.co.uk, Trackr API, Totaljobs, Gradcracker. "
        "python-jobspy integration for Indeed and Glassdoor. "
        "ATS-optimised PDF generation via pdfkit wkhtmltopdf, markdown-to-HTML pipeline. "
        "Telegram Bot API real-time push notifications. "
        "Python asyncio, pandas, REST API integration, SQLite application database, "
        "full-stack software engineering, agentic AI systems, machine learning pipelines."
    ),
}

projects = list(portfolio.keys())
enriched_descriptions = [f"{k.replace('_', ' ')}: {v}" for k, v in portfolio.items()]
project_embeddings = model.encode(enriched_descriptions)

# ── State ─────────────────────────────────────────────────────────────────────

class AppState(TypedDict):
    job_desc: str
    job_url: str
    company_name: str
    company_research: str
    program_type: str
    skills_required: str
    historical_context: str
    is_duplicate: bool
    cv_draft: str
    cover_letter: str
    feedback: Annotated[list[str], operator.add]
    loops: int
    triage_passed: bool
    triage_score: float
    retrieved_projects: str
    static_bio: str
    pdf_path: str

# ── LLM ───────────────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

HEADERS          = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0"}
CV_OUTPUT_DIR    = "cv_outputs"
DB_PATH          = "applications.db"
JOBS_CACHE_FILE  = "jobs_cache.json"
JOBS_CACHE_TTL   = 23 * 3600  # refresh after 23 hours

_EMOJI_RE = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            applied_at       TEXT NOT NULL,
            company          TEXT,
            role             TEXT,
            program_type     TEXT DEFAULT 'internship',
            job_url          TEXT,
            triage_score     REAL,
            skills_required  TEXT,
            pdf_path         TEXT,
            status           TEXT DEFAULT 'applied',
            interview_status TEXT DEFAULT 'none',
            interview_date   TEXT,
            outcome          TEXT,
            notes            TEXT
        )
    """)
    # Migrate older tables missing new columns
    for col, typedef in [
        ("program_type",     "TEXT DEFAULT 'internship'"),
        ("skills_required",  "TEXT"),
        ("interview_status", "TEXT DEFAULT 'none'"),
        ("interview_date",   "TEXT"),
        ("outcome",          "TEXT"),
        ("notes",            "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    conn.commit()
    conn.close()

# ── Scraper helpers ───────────────────────────────────────────────────────────

# URLs returned to the agent this session — prevents re-scraping the same job after a triage fail
_session_tried: set  = set()
# JobSpy results pre-fetched once at startup and consumed across retry attempts
_jobspy_pool:   list = []

def _url_key(url: str) -> str:
    """Stable dedup key. Reed appends a changing sid= session token — strip it. For
    Glassdoor/Indeed the query param IS the job ID, so keep the full URL."""
    if not url:
        return ""
    if "reed.co.uk" in url:
        parsed = urlparse(url)
        return f"{parsed.netloc}{parsed.path}"
    return url

_INTERN_TERMS = ("intern", "placement", "spring week", "insight week", "graduate", "junior", "student", "trainee")
_AUTH_WALLED  = ("linkedin.com", "glassdoor.com")  # jobspy already fetches full descriptions for these

# ── Trackr API queries (UK early-careers, curated, structured) ───────────────
# (params dict, program_type label)
# Summer internships and part-time only — no graduate schemes, placements, or spring weeks
TRACKR_QUERIES = [
    ({"region": "UK", "industry": "Engineering", "season": "2027", "type": "summer-internships"}, "summer internship"),
    ({"region": "UK", "industry": "Engineering", "season": "2026", "type": "summer-internships"}, "summer internship"),
]

# ── JobSpy search terms (Indeed + Glassdoor — LinkedIn rate-limits heavily) ──
# (search_term, [sites], program_type)
JOBSPY_SEARCHES = [
    ("robotics engineer summer internship",        ["indeed", "glassdoor"], "summer internship"),
    ("embedded systems engineer intern",           ["indeed", "glassdoor"], "summer internship"),
    ("autonomous systems summer internship",       ["indeed", "glassdoor"], "summer internship"),
    ("mechatronics engineering internship",        ["indeed"],              "summer internship"),
    ("control systems engineer intern",            ["indeed", "glassdoor"], "summer internship"),
    ("aerospace engineering summer intern",        ["indeed", "glassdoor"], "summer internship"),
    ("software engineer robotics intern",          ["indeed"],              "summer internship"),
    ("c++ engineer summer internship",             ["indeed", "glassdoor"], "summer internship"),
    ("robotics part time student engineer",        ["indeed"],              "part time"),
    ("embedded part time student engineer",        ["indeed"],              "part time"),
]

# ── Reed search URLs ─────────────────────────────────────────────────────────
REED_SEARCHES = [
    # Summer internships — robotics / ROS / autonomous
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=robotics+engineer&location=united+kingdom",          "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=ros+robotics+intern&location=united+kingdom",        "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=autonomous+systems+intern&location=united+kingdom",  "summer internship"),
    # Summer internships — embedded / firmware / hardware
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=embedded+systems+engineer&location=united+kingdom",  "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=embedded+firmware+intern&location=united+kingdom",   "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=mechatronics+intern&location=united+kingdom",        "summer internship"),
    # Summer internships — control / simulation / aerospace
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=control+systems+intern&location=united+kingdom",     "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=simulation+engineer+intern&location=united+kingdom", "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=aerospace+engineering+intern&location=united+kingdom","summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=software+engineer+robotics&location=united+kingdom", "summer internship"),
    ("https://www.reed.co.uk/jobs/internship-jobs?keywords=c%2B%2B+engineer+intern&location=united+kingdom",   "summer internship"),
    # Part-time student roles
    ("https://www.reed.co.uk/jobs/part-time-jobs?keywords=robotics+student+engineer",                           "part time"),
    ("https://www.reed.co.uk/jobs/part-time-jobs?keywords=embedded+engineer+student",                           "part time"),
]

GRADCRACKER_SEARCHES = [
    ("https://www.gradcracker.com/search/electrical-electronic-engineering/all/jobs?keyword=robotics",      "summer internship"),
    ("https://www.gradcracker.com/search/engineering-it/all/jobs?keyword=embedded+systems",                 "summer internship"),
    ("https://www.gradcracker.com/search/engineering-it/all/jobs?keyword=autonomous+systems",               "summer internship"),
]

# ── Totaljobs search URLs ─────────────────────────────────────────────────────
TOTALJOBS_SEARCHES = [
    ("https://www.totaljobs.com/jobs/internship?keywords=robotics+engineer",          "summer internship"),
    ("https://www.totaljobs.com/jobs/internship?keywords=embedded+systems+engineer",  "summer internship"),
    ("https://www.totaljobs.com/jobs/internship?keywords=mechatronics",               "summer internship"),
    ("https://www.totaljobs.com/jobs/internship?keywords=control+systems+engineer",   "summer internship"),
]

def _already_applied(job_url: str, company: str, role: str) -> bool:
    """Return True if this job is already in the DB (dedup check)."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        if job_url:
            key = _url_key(job_url)
            rows = conn.execute("SELECT job_url FROM applications").fetchall()
            if any(_url_key(r[0]) == key for r in rows if r[0]):
                conn.close()
                return True
        if conn.execute(
            "SELECT 1 FROM applications WHERE company = ? AND role = ? AND applied_at > datetime('now', '-30 days')",
            (company, role[:120]),
        ).fetchone():
            conn.close()
            return True
        conn.close()
    except Exception:
        pass
    return False


def _parse_reed_page(url: str, program_type: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = soup.find_all("article", attrs={"data-qa": "job-result"}) or soup.find_all("article")
    for article in articles[:10]:
        title_el = article.find("h2") or article.find("h3")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if not any(k in title.lower() for k in _INTERN_TERMS):
            continue

        link = title_el.find("a") or article.find("a", href=True)
        job_url = ""
        if link and link.get("href"):
            href = link["href"]
            job_url = f"https://www.reed.co.uk{href}" if href.startswith("/") else href

        company_el = (
            article.find(attrs={"data-qa": "job-result-company-name"})
            or article.find(class_=re.compile(r"employer|company", re.I))
        )
        company_name = company_el.get_text(strip=True) if company_el else "Unknown Company"

        desc_el = (
            article.find(attrs={"data-qa": "job-result-description"})
            or article.find(class_=re.compile(r"description", re.I))
        )
        snippet = desc_el.get_text(strip=True) if desc_el else ""

        key = _url_key(job_url)
        if key in _session_tried or _already_applied(job_url, company_name, title):
            print(f"  Skipping (tried/applied): {title} @ {company_name}")
            continue

        _session_tried.add(key)
        print(f"Scraped (Reed): {title} @ {company_name} [{program_type}]")
        return {
            "job_desc":     f"Role: {title}\nCompany: {company_name}\n\n{snippet}",
            "job_url":      job_url,
            "company_name": company_name,
            "program_type": program_type,
        }
    return None


def _build_trackr_entry(prog: dict, program_type: str) -> dict:
    """Convert a Trackr API programme dict into the standard job entry format."""
    company_data = prog.get("company") or {}
    company      = company_data.get("name") or prog.get("companyId", "Unknown Company")
    url          = prog.get("url") or ""
    company_desc = (company_data.get("description") or "")[:600]
    locations    = ", ".join(loc.split("|")[-1] for loc in (prog.get("locations") or []) if loc) or "UK"
    disciplines  = ", ".join(prog.get("disciplines") or [])
    eligibility  = prog.get("eligibility") or ""
    categories   = ", ".join(prog.get("categories") or [])

    job_desc = "\n\n".join(filter(None, [
        f"Role: {prog.get('name', '')}",
        f"Company: {company}",
        f"Type: {program_type} | Location: {locations}",
        f"Sector: {categories}" if categories else "",
        company_desc,
        f"Disciplines: {disciplines}" if disciplines else "",
        f"Eligibility: {eligibility}" if eligibility else "",
    ]))

    return {
        "id":           f"trackr:{prog.get('id', '')}",
        "title":        prog.get("name", ""),
        "company":      company,
        "job_url":      url or f"https://app.the-trackr.com/uk-engineering/{prog.get('type', '')}",
        "job_desc":     job_desc,
        "program_type": program_type,
    }


# Pre-scored Trackr pool — built at startup, sorted by triage score descending
_trackr_pool: list = []


def prefetch_trackr():
    """Fetch all Trackr engineering programmes, score against portfolio, sort best-first."""
    global _trackr_pool
    session_id = os.getenv("TRACKR_SESSION_ID", "")
    if not session_id:
        print("TRACKR_SESSION_ID not set — skipping Trackr.")
        return

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Origin": "https://app.the-trackr.com",
        "Referer": "https://app.the-trackr.com/",
        "sessionid": session_id,
    }

    raw = []
    for params, program_type in TRACKR_QUERIES:
        try:
            resp = requests.get("https://api.the-trackr.com/programmes",
                                params=params, headers=headers, timeout=10)
            if resp.status_code == 401:
                print("Trackr: session expired — update TRACKR_SESSION_ID in .env")
                return
            resp.raise_for_status()
            for prog in resp.json().get("programmes", []):
                raw.append(_build_trackr_entry(prog, program_type))
        except Exception as e:
            print(f"Trackr prefetch ({params.get('type', '')}): {e}")

    if not raw:
        return

    # Score every entry against portfolio embeddings and sort highest-first
    descs   = [e["job_desc"] for e in raw]
    q_embs  = model.encode(descs, batch_size=64, show_progress_bar=False)
    scores  = cosine_similarity(q_embs, project_embeddings).max(axis=1)  # best project match per job
    ranked  = sorted(zip(scores, raw), key=lambda x: x[0], reverse=True)
    _trackr_pool[:] = [entry for _, entry in ranked]
    print(f"Trackr prefetch: {len(_trackr_pool)} roles ranked (top score: {ranked[0][0]:.2f} — {ranked[0][1]['title']} @ {ranked[0][1]['company']}).")


def _scrape_trackr() -> dict | None:
    """Return the next unused entry from the pre-scored Trackr pool."""
    for entry in _trackr_pool:
        key = entry["id"]
        if key in _session_tried or _already_applied(entry["job_url"], entry["company"], entry["title"]):
            continue
        _session_tried.add(key)
        print(f"Scraped (Trackr): {entry['title']} @ {entry['company']} [{entry['program_type']}]")
        return {
            "job_desc":     entry["job_desc"],
            "job_url":      entry["job_url"],
            "company_name": entry["company"],
            "program_type": entry["program_type"],
        }
    return None


def _parse_totaljobs_page(url: str, program_type: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = (
        soup.find_all("article", attrs={"data-jobid": True})
        or soup.find_all(class_=re.compile(r"job-listing|jobResult", re.I))
        or soup.find_all("article")
    )
    for article in articles[:10]:
        title_el = article.find(["h2", "h3"]) or article.find(class_=re.compile(r"job-title", re.I))
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if not any(k in title.lower() for k in _INTERN_TERMS):
            continue

        link = title_el.find("a") or article.find("a", href=True)
        job_url = ""
        if link and link.get("href"):
            href = link["href"]
            job_url = f"https://www.totaljobs.com{href}" if href.startswith("/") else href

        company_el = article.find(class_=re.compile(r"company|employer", re.I))
        company_name = company_el.get_text(strip=True) if company_el else "Unknown Company"

        desc_el = article.find(class_=re.compile(r"job-description|description|summary", re.I))
        snippet = desc_el.get_text(strip=True) if desc_el else ""

        key = _url_key(job_url)
        if key in _session_tried or _already_applied(job_url, company_name, title):
            print(f"  Skipping (tried/applied): {title} @ {company_name}")
            continue

        _session_tried.add(key)
        print(f"Scraped (Totaljobs): {title} @ {company_name} [{program_type}]")
        return {
            "job_desc":     f"Role: {title}\nCompany: {company_name}\n\n{snippet}",
            "job_url":      job_url,
            "company_name": company_name,
            "program_type": program_type,
        }
    return None


def _parse_gradcracker_page(url: str, program_type: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Gradcracker job cards vary by section — try several selectors
    cards = (
        soup.find_all(class_=re.compile(r"vacancy|job-card|listing", re.I))
        or soup.find_all("article")
        or soup.find_all("li", class_=re.compile(r"job|vacancy", re.I))
    )
    for card in cards[:10]:
        title_el = card.find(["h2", "h3", "h4"]) or card.find(class_=re.compile(r"title|heading", re.I))
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if len(title) < 5:
            continue

        link = title_el.find("a") or card.find("a", href=True)
        job_url = ""
        if link and link.get("href"):
            href = link["href"]
            job_url = f"https://www.gradcracker.com{href}" if href.startswith("/") else href

        company_el = card.find(class_=re.compile(r"employer|company|recruiter", re.I))
        company_name = company_el.get_text(strip=True) if company_el else "Unknown Company"

        desc_el = card.find(class_=re.compile(r"description|summary|snippet", re.I))
        snippet = desc_el.get_text(strip=True) if desc_el else ""

        key = _url_key(job_url)
        if key in _session_tried or _already_applied(job_url, company_name, title):
            print(f"  Skipping (tried/applied): {title} @ {company_name}")
            continue

        _session_tried.add(key)
        print(f"Scraped (Gradcracker): {title} @ {company_name} [{program_type}]")
        return {
            "job_desc":     f"Role: {title}\nCompany: {company_name}\n\n{snippet}",
            "job_url":      job_url,
            "company_name": company_name,
            "program_type": program_type,
        }
    return None

# ── Nodes ─────────────────────────────────────────────────────────────────────

async def prefetch_jobspy():
    """Fetch jobs from Indeed + Glassdoor once per day, cache to disk to avoid rate limits."""
    global _jobspy_pool

    # Load from disk cache if still fresh
    if os.path.exists(JOBS_CACHE_FILE):
        with open(JOBS_CACHE_FILE) as f:
            cache = json.load(f)
        age = time.time() - cache.get("timestamp", 0)
        if age < JOBS_CACHE_TTL:
            _jobspy_pool = cache.get("jobs", [])
            print(f"JobSpy cache loaded: {len(_jobspy_pool)} jobs ({int(age/3600)}h old, refreshes in {int((JOBS_CACHE_TTL-age)/3600)}h).")
            return

    print("Fetching fresh jobs from Indeed and Glassdoor (this takes ~30s)...")
    fresh = []
    for search_term, sites, program_type in JOBSPY_SEARCHES:
        try:
            df = await asyncio.to_thread(
                _jobspy_scrape,
                site_name=sites,
                search_term=search_term,
                location="United Kingdom",
                results_wanted=10,
                hours_old=168,
                verbose=0,
            )
            for _, row in df.iterrows():
                title       = str(row.get("title", "") or "")
                company     = str(row.get("company", "") or "Unknown Company")
                job_url     = str(row.get("job_url", "") or "")
                description = str(row.get("description", "") or "")[:2000]
                direct_url  = str(row.get("job_url_direct", "") or "")
                site        = str(row.get("site", ""))

                if not any(k in title.lower() for k in _INTERN_TERMS):
                    continue
                if not title:
                    continue

                fetch_url = (
                    direct_url
                    if direct_url and not any(d in direct_url for d in _AUTH_WALLED)
                    else job_url
                )
                fresh.append({
                    "title": title, "company": company,
                    "job_url": fetch_url or job_url,
                    "description": description,
                    "site": site, "program_type": program_type,
                })
        except Exception as e:
            print(f"JobSpy ({', '.join(sites)}, '{search_term[:30]}'): {e}")

    _jobspy_pool = fresh
    with open(JOBS_CACHE_FILE, "w") as f:
        json.dump({"jobs": fresh, "timestamp": time.time()}, f)
    print(f"JobSpy prefetch complete: {len(fresh)} results cached to {JOBS_CACHE_FILE}.")


async def scrape_jobs(state: AppState):
    """
    Multi-source scraper — tried in priority order:
      1. Pre-fetched JobSpy pool (LinkedIn, Indeed, Glassdoor)
      2. Reed    : UK-specific, server-rendered
      3. Totaljobs: UK-specific, server-rendered
      4. Gradcracker: UK STEM students / spring weeks
    """

    # ── 1. Trackr (structured UK early-careers API) ──────────────────────────
    result = _scrape_trackr()
    if result:
        return result

    # ── 2. Pre-fetched JobSpy pool ────────────────────────────────────────────
    for job in _jobspy_pool:
        title   = job["title"]
        company = job["company"]
        job_url = job["job_url"]

        key = _url_key(job_url)
        if key in _session_tried or _already_applied(job_url, company, title):
            continue

        _session_tried.add(key)
        print(f"Scraped ({job['site']}): {title} @ {company} [{job['program_type']}]")
        return {
            "job_desc":     f"Role: {title}\nCompany: {company}\n\n{job['description']}",
            "job_url":      job_url,
            "company_name": company,
            "program_type": job["program_type"],
        }

    # ── 2. Reed ───────────────────────────────────────────────────────────────
    for url, program_type in REED_SEARCHES:
        try:
            result = _parse_reed_page(url, program_type)
            if result:
                return result
        except Exception as e:
            print(f"Reed ({url[:55]}): {e}")

    # ── 3. Totaljobs ──────────────────────────────────────────────────────────
    for url, program_type in TOTALJOBS_SEARCHES:
        try:
            result = _parse_totaljobs_page(url, program_type)
            if result:
                return result
        except Exception as e:
            print(f"Totaljobs ({url[:55]}): {e}")

    # ── 4. Gradcracker ────────────────────────────────────────────────────────
    for url, program_type in GRADCRACKER_SEARCHES:
        try:
            result = _parse_gradcracker_page(url, program_type)
            if result:
                return result
        except Exception as e:
            print(f"Gradcracker: {e}")

    print("All scrapers exhausted. Using fallback.")
    return {
        "job_desc":     "Role: Robotics Engineer Intern\nCompany: Unknown Company\n\nSeeking robotics engineer intern with ROS 2, simulation experience, and C++ control loops.",
        "job_url":      "",
        "company_name": "Unknown Company",
        "program_type": "summer internship",
    }


async def check_history(state: AppState):
    """Deduplicate against DB and build a learning context string from past interview outcomes."""
    init_db()
    conn = sqlite3.connect(DB_PATH)

    company  = state.get("company_name", "")
    job_url  = state.get("job_url", "")
    job_desc = state.get("job_desc", "")
    role     = job_desc.split("\n")[0].replace("Role:", "").strip()[:120]

    # Deduplicate by URL first (normalised), then company+role within 30 days
    if job_url:
        key = _url_key(job_url)
        all_urls = conn.execute("SELECT job_url FROM applications").fetchall()
        dup = any(_url_key(r[0]) == key for r in all_urls if r[0])
        if dup:
            print(f"Duplicate (URL match, id={dup[0]}). Skipping.")
            conn.close()
            return {"is_duplicate": True, "historical_context": ""}

    dup = conn.execute(
        "SELECT id FROM applications WHERE company = ? AND role = ? AND applied_at > datetime('now', '-30 days')",
        (company, role),
    ).fetchone()
    if dup:
        print(f"Duplicate (company+role within 30 days, id={dup[0]}). Skipping.")
        conn.close()
        return {"is_duplicate": True, "historical_context": ""}

    # Build learning context from past outcomes
    total      = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    interviews = conn.execute("SELECT COUNT(*) FROM applications WHERE interview_status != 'none'").fetchone()[0]
    lines = [f"AGENT HISTORY: {total} applications sent, {interviews} progressed to interview."]

    if interviews > 0:
        rows = conn.execute(
            "SELECT skills_required FROM applications WHERE interview_status != 'none' AND skills_required IS NOT NULL"
        ).fetchall()
        if rows:
            skill_blob = " | ".join(r[0] for r in rows if r[0])[:400]
            lines.append(f"Skills in interview-winning applications: {skill_blob}")
            lines.append("Prioritise mirroring these skills in the CV if they appear in this job description.")

    rejected = conn.execute(
        "SELECT company FROM applications WHERE outcome = 'rejected' ORDER BY applied_at DESC LIMIT 5"
    ).fetchall()
    if rejected:
        lines.append(f"Recent rejections (avoid pattern-matching these): {', '.join(r[0] for r in rejected)}")

    conn.close()
    context = "\n".join(lines)
    print(f"History loaded: {total} applications, {interviews} interviews.")
    return {"is_duplicate": False, "historical_context": context}


def route_after_history(state: AppState):
    return END if state.get("is_duplicate") else "research_company"


async def research_company(state: AppState):
    """Fetch the full job page for Reed/Totaljobs. Skip for LinkedIn/Glassdoor — jobspy already returned the full description."""
    job_url      = state.get("job_url", "")
    company_name = state.get("company_name", "Unknown Company")
    base_desc    = state.get("job_desc", "")

    # LinkedIn/Glassdoor require auth (jobspy already fetched description)
    # Trackr app URLs are React SPAs — nothing to fetch, company desc already in job_desc
    skip_domains = (*_AUTH_WALLED, "app.the-trackr.com")
    if any(domain in job_url for domain in skip_domains) and len(base_desc) > 100:
        print(f"Skipping page fetch for {job_url[:50]} — using existing description.")
        return {"company_research": base_desc[:3000]}

    if not job_url:
        return {"company_research": f"{company_name} — no URL available."}

    try:
        resp = requests.get(job_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Company name extraction
        if company_name == "Unknown Company":
            employer_el = (
                soup.find(attrs={"data-qa": "job-header-company-name"})
                or soup.find(class_=re.compile(r"recruiter-?link|employer-?name|company-?name|posted-?by", re.I))
            )
            if employer_el:
                raw = employer_el.get_text(strip=True)
                by_match = re.search(r'by\s*([A-Z][a-z].+)$', raw)
                extracted = by_match.group(1).strip() if by_match else raw
                if extracted and len(extracted) < 80:
                    company_name = extracted
                    print(f"Company name: {company_name}")

        # Full description
        full = (
            soup.find(attrs={"itemprop": "description"})
            or soup.find(id=re.compile(r"job-description", re.I))
            or soup.find(class_=re.compile(r"job-description|description-content", re.I))
        )
        if full:
            research = full.get_text(separator="\n", strip=True)[:3000]
            print(f"Research fetched ({len(research)} chars).")
            return {
                "company_research": research,
                "job_desc":         f"{base_desc}\n\n{research}",
                "company_name":     company_name,
            }
    except Exception as e:
        print(f"Research failed: {e}")

    return {"company_research": f"{company_name} — full description not available.", "company_name": company_name}


async def triage_job(state: AppState):
    job    = state.get("job_desc", "")
    q_emb  = model.encode([job])
    scores = cosine_similarity(q_emb, project_embeddings)[0]
    top2   = np.argsort(scores)[-2:][::-1]
    best   = float(scores[top2[0]])

    if best >= 0.32:
        retrieved = "\n\n".join(
            f"PROJECT — {projects[i]}:\n{enriched_descriptions[i]}" for i in top2
        )
        print(f"Triage passed ({best:.2f}). Projects: {[projects[i] for i in top2]}")
        return {"triage_passed": True,  "retrieved_projects": retrieved, "triage_score": best}

    print(f"Triage failed ({best:.2f} < 0.32).")
    return {"triage_passed": False, "retrieved_projects": "", "triage_score": best}


def route_triage(state: AppState):
    return "extract_skills" if state.get("triage_passed") else END


async def extract_skills(state: AppState):
    """LLM extracts a concise comma-separated list of required technical skills."""
    job = state.get("job_desc", "")
    if len(job) < 100:
        print("Skills: (description too short to extract)")
        return {"skills_required": ""}

    prompt = (
        "Extract the required technical skills from this job description as a concise comma-separated list.\n"
        "Include only concrete technical skills: languages, frameworks, tools, hardware, protocols.\n"
        "No soft skills. Maximum 15 items. One line only. If no skills are clear, return 'general engineering'.\n\n"
        f"Job description:\n{job[:1500]}"
    )
    response = await llm.ainvoke(prompt)
    skills = response.content.strip().replace("\n", ", ")[:400]
    print(f"Skills: {skills}")
    return {"skills_required": skills}


async def draft_cv(state: AppState):
    loops   = state.get("loops", 0) + 1
    bio     = state.get("static_bio", "")
    proj    = state.get("retrieved_projects", "")
    job     = state.get("job_desc", "")
    company = state.get("company_name", "the company")
    research = state.get("company_research", "")
    history = state.get("historical_context", "")
    feedback = state.get("feedback", [])
    prev_fb  = feedback[-1] if feedback else ""

    history_block  = f"\nAGENT LEARNING CONTEXT (use to prioritise):\n{history}\n" if history else ""
    revision_block = f"\nPREVIOUS REVIEWER FEEDBACK TO FIX:\n{prev_fb}\n" if prev_fb else ""

    prompt = f"""You are a senior technical recruiter writing a CV for an internship applicant.

JOB DESCRIPTION:
{job}

FULL ROLE CONTEXT:
{research}

CANDIDATE BIO:
{bio}

RELEVANT PROJECTS (use ONLY these — do not invent others):
{proj}
{history_block}{revision_block}
Write an ATS-optimised CV in Markdown. Strict rules:
- ZERO emojis or non-ASCII characters. Plain ASCII text only.
- Mirror exact keywords from the job description in skills and bullets.
- Every bullet: Google XYZ formula — "Accomplished [X] as measured by [Y] by doing [Z]".
- Skills section: every technology from the job description the candidate demonstrably has.
- Concise — one page equivalent.
- Do NOT fabricate metrics or experience not in the bio or projects."""

    response = await llm.ainvoke(prompt)
    return {"cv_draft": response.content, "loops": loops}


async def review_draft(state: AppState):
    loops = state.get("loops", 0)
    if loops >= 2:
        return {"feedback": ["APPROVED — max revision iterations reached."]}

    draft = state.get("cv_draft", "")
    job   = state.get("job_desc", "")

    prompt = f"""You are a ruthless senior technical recruiter reviewing an internship CV.

JOB DESCRIPTION:
{job}

CV DRAFT:
{draft}

Check ALL four criteria:
1. KEYWORD COVERAGE: Required skills from the job description absent from the CV.
2. XYZ FORMULA: Quote any bullet not following "Accomplished X as measured by Y by doing Z".
3. ATS ISSUES: Any emoji, non-ASCII character, or markdown that breaks ATS text parsing.
4. FABRICATION: Any claim not supported by the listed projects or bio.

Start your response with exactly "APPROVED" if every criterion passes.
Otherwise start with "REVISION NEEDED:" and give numbered, specific instructions. No general praise."""

    response = await llm.ainvoke(prompt)
    return {"feedback": [response.content]}


def route_next(state: AppState):
    last = state.get("feedback", [""])[-1].strip()
    return "draft_cover_letter" if last.upper().startswith("APPROVED") else "drafter"


async def draft_cover_letter(state: AppState):
    bio      = state.get("static_bio", "")
    proj     = state.get("retrieved_projects", "")
    job      = state.get("job_desc", "")
    company  = state.get("company_name", "the company")
    research = state.get("company_research", "")
    prog     = state.get("program_type", "internship")

    prompt = f"""Write a professional cover letter for a {prog} application.

COMPANY: {company}
JOB DESCRIPTION: {job}
COMPANY CONTEXT: {research}
CANDIDATE BIO: {bio}
RELEVANT PROJECTS: {proj}

Rules:
- Exactly 3 paragraphs, 250-320 words total.
- Paragraph 1: Why THIS specific company — reference something concrete about their work.
- Paragraph 2: Your single most relevant project — what you built and the outcome (XYZ formula).
- Paragraph 3: What you will contribute and what you want to learn from this placement.
- Professional British English. Zero emojis. Zero special characters. Plain ASCII only.
- End with: Yours sincerely, Aryan Naik
- Do NOT invent experience not in the bio or projects."""

    response = await llm.ainvoke(prompt)
    return {"cover_letter": response.content}


async def compile_pdf(state: AppState):
    print("\n>>> COMPILING APPLICATION PDF... <<<")
    os.makedirs(CV_OUTPUT_DIR, exist_ok=True)

    cv_text  = strip_emojis(state.get("cv_draft", ""))
    cl_text  = strip_emojis(state.get("cover_letter", ""))
    slug     = re.sub(r"[^a-z0-9]+", "_", state.get("company_name", "application").lower()).strip("_")
    date_str = datetime.now().strftime("%Y%m%d")

    css = """
    <style>
        body  { font-family: Arial, sans-serif; font-size: 11pt; color: #000; line-height: 1.45; margin: 0.75in; }
        h1    { font-size: 17pt; text-align: center; border-bottom: 2px solid #000; margin-bottom: 10px; padding-bottom: 5px; }
        h2    { font-size: 12pt; border-bottom: 1px solid #aaa; margin-top: 18px; margin-bottom: 8px; padding-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
        h3    { font-size: 11pt; margin-bottom: 2px; }
        p, li { margin-top: 2px; margin-bottom: 4px; }
        hr    { display: none; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
        td, th { padding: 3px 8px; border: 1px solid #ccc; font-size: 10.5pt; }
        .page-break { page-break-before: always; }
    </style>
    """

    cv_html  = markdown.markdown(cv_text, extensions=["tables"])
    cl_html  = markdown.markdown(cl_text, extensions=["tables"])
    full_html = (
        f"<html><head><meta charset='utf-8'>{css}</head><body>"
        f"{cv_html}"
        f'<div class="page-break"></div>'
        f"{cl_html}"
        f"</body></html>"
    )

    output_filename = os.path.join(CV_OUTPUT_DIR, f"{slug}_{date_str}.pdf")
    options = {
        "page-size": "A4", "margin-top": "0", "margin-right": "0",
        "margin-bottom": "0", "margin-left": "0",
        "encoding": "UTF-8", "disable-smart-shrinking": "",
    }

    try:
        pdfkit.from_string(full_html, output_filename, options=options)
        print(f">>> PDF COMPILED: {os.path.abspath(output_filename)} <<<")
    except Exception as e:
        print(f">>> PDF COMPILATION FAILED: {e} <<<")
        output_filename = ""

    return {"pdf_path": output_filename}


async def log_application(state: AppState):

    init_db()
    job_desc = state.get("job_desc", "")
    role     = job_desc.split("\n")[0].replace("Role:", "").strip()[:120]

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO applications
           (applied_at, company, role, program_type, job_url, triage_score, skills_required, pdf_path, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            state.get("company_name", ""),
            role,
            state.get("program_type", "internship"),
            state.get("job_url", ""),
            state.get("triage_score", 0.0),
            state.get("skills_required", ""),
            state.get("pdf_path", ""),
            "applied",
        ),
    )
    conn.commit()
    conn.close()
    print(f">>> Logged to {os.path.abspath(DB_PATH)} <<<")
    return {}


# ── Graph wiring ──────────────────────────────────────────────────────────────

workflow = StateGraph(AppState)
workflow.add_node("scraper",            scrape_jobs)
workflow.add_node("check_history",      check_history)
workflow.add_node("research_company",   research_company)
workflow.add_node("triage",             triage_job)
workflow.add_node("extract_skills",     extract_skills)
workflow.add_node("drafter",            draft_cv)
workflow.add_node("reviewer",           review_draft)
workflow.add_node("draft_cover_letter", draft_cover_letter)
workflow.add_node("compile_pdf",        compile_pdf)
workflow.add_node("log_application",    log_application)

workflow.set_entry_point("scraper")
workflow.add_edge("scraper",            "check_history")
workflow.add_conditional_edges("check_history", route_after_history, {END: END, "research_company": "research_company"})
workflow.add_edge("research_company",   "triage")
workflow.add_conditional_edges("triage", route_triage, {END: END, "extract_skills": "extract_skills"})
workflow.add_edge("extract_skills",     "drafter")
workflow.add_edge("drafter",            "reviewer")
workflow.add_conditional_edges(
    "reviewer", route_next,
    {"draft_cover_letter": "draft_cover_letter", "drafter": "drafter"},
)
workflow.add_edge("draft_cover_letter", "compile_pdf")
workflow.add_edge("compile_pdf",        "log_application")
workflow.add_edge("log_application",    END)

app = workflow.compile()


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram_notification(company: str, program: str, pdf_path: str, job_url: str):
    """Notify that a CV is ready in cv_outputs/ — no approval needed, just a heads-up."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("Telegram credentials missing. Skipping notification.")
        return

    # HTML parse mode — avoids _ and & in URLs breaking Telegram's Markdown parser
    message = (
        "\U0001f916 <b>JOB DROP — CV Ready</b>\n\n"
        f"<b>Company:</b> {company}\n"
        f"<b>Program:</b> {program}\n"
        f"<b>File:</b> <code>{os.path.basename(pdf_path)}</code>\n\n"
        f"Apply at: {job_url or 'check cv_outputs/ folder'}\n\n"
        "<i>CV and cover letter saved. Apply manually when ready.</i>"
    )

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"Telegram notification sent (message_id: {resp.json()['result']['message_id']})")
    except Exception as e:
        print(f"Telegram notification failed ({e}).")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    base_state = {
        "loops":              0,
        "feedback":           [],
        "static_bio": (
            "Name: Aryan Naik | Email: aryannaik1205@gmail.com | "
            "GitHub: github.com/RobotsAndAryan | "
            "Education: 2nd-year MEng Robotics and Artificial Intelligence at "
            "University College London (UCL) | Graduation: 2029."
        ),
        "job_desc":           "",
        "job_url":            "",
        "company_name":       "",
        "company_research":   "",
        "program_type":       "",
        "skills_required":    "",
        "historical_context": "",
        "is_duplicate":       False,
        "cv_draft":           "",
        "cover_letter":       "",
        "triage_passed":      False,
        "triage_score":       0.0,
        "retrieved_projects": "",
        "pdf_path":           "",
    }

    print("Prefetching Trackr (UK engineering roles, pre-scored)...")
    prefetch_trackr()

    print("Prefetching jobs from LinkedIn, Indeed, Glassdoor...")
    await prefetch_jobspy()

    generated = 0
    for attempt in range(1, 9):
        print(f"\n--- Attempt {attempt}/8 ---")

        vals = await app.ainvoke(dict(base_state))

        if vals.get("is_duplicate"):
            continue
        if not vals.get("triage_passed"):
            print(f"Triage failed ({vals.get('triage_score', 0):.2f}). Trying next job...")
            continue

        pdf     = vals.get("pdf_path", "")
        company = vals.get("company_name", "")
        job_url = vals.get("job_url", "")
        program = vals.get("program_type", "")
        generated += 1

        print(f"\n{'='*60}")
        print(f"  CV READY [{generated}]: {os.path.abspath(pdf) if pdf else 'compilation failed'}")
        print(f"  Company : {company}")
        print(f"  Program : {program}")
        print(f"  Apply   : {job_url or 'check cv_outputs/ folder'}")
        print(f"{'='*60}\n")

        send_telegram_notification(company, program, pdf, job_url)
        # Continue — don't stop at 1, keep generating for all 8 attempts

    if generated == 0:
        print("No suitable job found after 8 attempts. Try running again later.")
    else:
        print(f"\nDone. {generated} CV(s) generated in cv_outputs/. Apply manually.")

if __name__ == "__main__":
    asyncio.run(main())
