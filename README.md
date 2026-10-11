# Job Drop Agent

Automated internship application pipeline. Scrapes live listings, generates a tailored CV and cover letter per role, compiles a 2-page PDF, and notifies you on Telegram. You review the PDF and apply manually.

---

## Setup

```bash
# Activate the virtual environment (always do this first)
cd ~/Documents/Home/Career/j_agent
source venv/bin/activate
```

Required `.env` variables:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

TRACKR_SESSION_ID=...   # refresh every few weeks (see Maintenance)
```

---

## Running the agent

### Auto mode — scrapes all sources automatically

```bash
python3 state_machine.py
```

Runs up to 8 attempts. Each attempt finds the next most-relevant role (Trackr → Reed → Totaljobs → Indeed/Glassdoor), generates a tailored CV + cover letter, saves a PDF to `cv_outputs/`, and sends a Telegram notification. Never applies to the same company twice.

### Manual mode — you found a job yourself

```bash
python3 state_machine.py --manual
```

Use this when you find a job on LinkedIn, a company website, or anywhere else the scraper doesn't cover.

You'll be prompted:

```
Company name: Boston Dynamics
Job title   : Robotics Software Engineering Intern
Job URL     : https://bostondynamics.com/careers/123
Program type [summer internship]: summer internship

Paste the full job description below.
Type END on its own line when done:

We are looking for a robotics software intern...
[paste the full text from LinkedIn / company site]
END
```

The rest is identical — triage, tailored CV, cover letter, PDF, Telegram notification, logged in the database.

**Tip:** paste as much of the job description as possible — the more text, the better the CV mirrors the recruiter's exact keywords.

---

## Tracking applications

```bash
# Show all applications (most recent first)
python3 db_viewer.py

# Show only roles that progressed to interview
python3 db_viewer.py --filter interview

# Full detail for application ID 5 (skills required, URL, PDF path)
python3 db_viewer.py --detail 5

# Statistics: interview rate, average triage score, top skills
python3 db_viewer.py --stats

# Update interview status when a company replies
python3 db_viewer.py --update 5 phone_screen
python3 db_viewer.py --update 5 phone_screen --date 2026-11-01 --notes "Recruiter called, liked AEGIS project"
python3 db_viewer.py --update 5 rejected --outcome rejected
python3 db_viewer.py --update 5 offer --outcome offer
```

**Interview status values:** `none` → `phone_screen` → `technical` → `final_round` → `offer` / `rejected`

As you log interview outcomes, the agent automatically prioritises the skills that got you interviews in future CVs.

---

## Output files

| File | What it is |
|---|---|
| `cv_outputs/{company}_{date}.pdf` | 2-page PDF: CV (page 1) + cover letter (page 2) |
| `applications.db` | SQLite log of every application — view with `db_viewer.py` |
| `jobs_cache.json` | 24-hour cache of Indeed/Glassdoor results — deleted automatically when stale |

---

## Recommended schedule

| Period | Action |
|---|---|
| **Now → December** | Run every morning. Most 2027 Trackr roles close Dec–Jan. Submit each PDF the same day. |
| **December → February** | Run every 2–3 days. Big companies (Boeing, BAE, Rolls-Royce) fill on a rolling basis. |
| **February onwards** | Run weekly. Catch late-opening roles and new listings. |

---

## Reset (start fresh)

```bash
rm -f jobs_cache.json
rm -f cv_outputs/*.pdf
python3 -c "import sqlite3; conn=sqlite3.connect('applications.db'); conn.execute('DELETE FROM applications'); conn.commit()"
```

---

## Maintenance

### Trackr session expired

When you see `"Trackr: session expired"` in the terminal:

1. Open `https://app.the-trackr.com` in Chrome and log in
2. Press `F12` → **Network** tab → filter by `api.the-trackr.com`
3. Click any request → **Headers** → copy the `sessionid` value
4. Update `.env`: `TRACKR_SESSION_ID="new-value-here"`

### Force a fresh job fetch (Indeed/Glassdoor)

```bash
rm -f jobs_cache.json
```

The cache refreshes automatically every 23 hours, but delete it manually if you want an immediate refresh.

---

## How it works

```
Trackr API (1949 roles, pre-scored)
Reed / Totaljobs / Gradcracker (BS4 scrapers)     ──→  scrape_jobs
Indeed / Glassdoor (python-jobspy, cached 23h)

          ↓
     check_history          deduplication + URL open/closed check
          ↓
    research_company         fetch full job page for richer description
          ↓
      triage_job             cosine similarity against portfolio (threshold 0.32)
          ↓
    extract_skills           LLM extracts required skills for the DB log
          ↓
      draft_cv ←──────┐      LLM writes ATS-optimised CV, mirroring job keywords
          ↓            │
     review_draft      │      LLM reviews against 4 criteria; loops if needed
          ↓            │
   APPROVED? ──No──────┘
          ↓ Yes
  draft_cover_letter         LLM writes 3-paragraph company-specific cover letter
          ↓
     compile_pdf              Markdown → HTML → A4 PDF (CV page 1, cover letter page 2)
          ↓
    log_application           Records company, role, skills, score, PDF path in SQLite
          ↓
Telegram notification + cv_outputs/{company}_{date}.pdf
```

**Sources checked in order:**
1. **Trackr** — 204 UK Engineering summer internships, pre-scored and ranked at startup
2. **Reed.co.uk** — UK-specific, 13 robotics/embedded search queries
3. **Totaljobs** — UK-specific fallback
4. **Gradcracker** — UK STEM student internships
5. **Indeed + Glassdoor** — via python-jobspy, cached daily

**Duplicate prevention:** three independent layers — in-session memory (`_session_tried`), DB URL lookup, and DB company+role lookup (30-day window).

**Closed job detection:** Trackr roles with a past `closingDate` are filtered at startup. All other URLs are checked with a HEAD + GET scan for closed-vacancy phrases before any CV is generated.
