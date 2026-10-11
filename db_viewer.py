#!/usr/bin/env python3
"""
Job application tracker — view and update applications.db

Usage:
  python3 db_viewer.py                       # show all applications
  python3 db_viewer.py --detail 3            # full detail for application id=3
  python3 db_viewer.py --filter interview    # show only apps that got interviews
  python3 db_viewer.py --stats               # success rates and skill analysis
  python3 db_viewer.py --update 3 phone_screen               # set interview status
  python3 db_viewer.py --update 3 phone_screen --date 2026-10-20
  python3 db_viewer.py --update 3 rejected --outcome rejected --notes "No response after 2 weeks"

Interview status values:
  none | phone_screen | technical | final_round | offer | rejected
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter

import pandas as pd

DB_PATH = "applications.db"

INTERVIEW_STATUSES = ("none", "phone_screen", "technical", "final_round", "offer", "rejected")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _col_widths(headers, rows):
    widths = list(map(len, headers))
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell) if cell is not None else ""))
    return widths


def _print_table(headers, rows):
    if not rows:
        print("  (no results)")
        return
    widths = _col_widths(headers, rows)
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    def fmt_row(r):
        return "| " + " | ".join(str(c if c is not None else "").ljust(widths[i]) for i, c in enumerate(r)) + " |"
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)
    print(f"  {len(rows)} row(s)")


def _truncate(text, n=40):
    if text is None:
        return ""
    return (str(text)[:n - 3] + "...") if len(str(text)) > n else str(text)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(filter_status=None):
    conn = sqlite3.connect(DB_PATH)
    if filter_status == "interview":
        query = "SELECT id, substr(applied_at,1,10) as date, company, role, program_type, round(triage_score,2) as score, interview_status FROM applications WHERE interview_status != 'none' ORDER BY id DESC"
    elif filter_status:
        query = f"SELECT id, substr(applied_at,1,10) as date, company, role, program_type, round(triage_score,2) as score, interview_status FROM applications WHERE interview_status = '{filter_status}' ORDER BY id DESC"
    else:
        query = "SELECT id, substr(applied_at,1,10) as date, company, role, program_type, round(triage_score,2) as score, interview_status FROM applications ORDER BY id DESC"

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("  (no results)")
        return

    # Truncate long text columns for display
    df["company"] = df["company"].str[:25]
    df["role"]    = df["role"].str[:38]
    print(df.to_string(index=False))
    print(f"\n  {len(df)} row(s)")


def cmd_detail(app_id: int):
    conn = sqlite3.connect(DB_PATH)
    cols = [d[1] for d in conn.execute("PRAGMA table_info(applications)").fetchall()]
    row  = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    conn.close()

    if not row:
        print(f"No application with id={app_id}")
        return

    print(f"\n{'='*60}")
    print(f"  Application #{app_id}")
    print(f"{'='*60}")
    for col, val in zip(cols, row):
        if col == "skills_required" and val:
            print(f"  {'skills_required':<20}: {val}")
        elif col == "job_url" and val:
            print(f"  {'job_url':<20}: {val[:80]}...")
        elif col == "pdf_path" and val:
            print(f"  {'pdf_path':<20}: {os.path.abspath(val)}")
        else:
            print(f"  {col:<20}: {val}")
    print(f"{'='*60}\n")


def cmd_stats():
    conn = sqlite3.connect(DB_PATH)
    total      = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    interviews = conn.execute("SELECT COUNT(*) FROM applications WHERE interview_status != 'none'").fetchone()[0]
    offers     = conn.execute("SELECT COUNT(*) FROM applications WHERE interview_status = 'offer'").fetchone()[0]
    rejected   = conn.execute("SELECT COUNT(*) FROM applications WHERE outcome = 'rejected'").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"  APPLICATION STATISTICS")
    print(f"{'='*50}")
    print(f"  Total applications : {total}")
    print(f"  Got interview      : {interviews}  ({interviews/total*100:.0f}% of total)" if total else "  Got interview      : 0")
    print(f"  Offers received    : {offers}")
    print(f"  Rejections logged  : {rejected}")

    # By program type
    by_type = conn.execute(
        "SELECT program_type, COUNT(*) as n, SUM(CASE WHEN interview_status != 'none' THEN 1 ELSE 0 END) as i "
        "FROM applications GROUP BY program_type"
    ).fetchall()
    if by_type:
        print(f"\n  BY PROGRAM TYPE:")
        _print_table(["Program", "Apps", "Interviews"], by_type)

    # Score distribution
    avg = conn.execute("SELECT AVG(triage_score) FROM applications").fetchone()[0]
    avg_int = conn.execute(
        "SELECT AVG(triage_score) FROM applications WHERE interview_status != 'none'"
    ).fetchone()[0]
    if avg:
        print(f"\n  Average triage score (all)        : {avg:.3f}")
    if avg_int:
        print(f"  Average triage score (interviews)  : {avg_int:.3f}")

    # Skills in interview-winning applications
    skill_rows = conn.execute(
        "SELECT skills_required FROM applications WHERE interview_status != 'none' AND skills_required IS NOT NULL"
    ).fetchall()
    if skill_rows:
        all_skills = []
        for (s,) in skill_rows:
            if s:
                all_skills.extend(sk.strip() for sk in s.split(","))
        counter = Counter(all_skills).most_common(10)
        print(f"\n  TOP SKILLS IN INTERVIEW-WINNING APPLICATIONS:")
        for skill, count in counter:
            print(f"    {skill:<35} x{count}")

    conn.close()
    print(f"{'='*50}\n")


def cmd_update(app_id: int, interview_status: str, date: str = None, outcome: str = None, notes: str = None):
    if interview_status not in INTERVIEW_STATUSES:
        print(f"Invalid status '{interview_status}'. Choose from: {', '.join(INTERVIEW_STATUSES)}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT id FROM applications WHERE id = ?", (app_id,)).fetchone()
    if not existing:
        print(f"No application with id={app_id}")
        conn.close()
        return

    updates, params = ["interview_status = ?"], [interview_status]
    if date:
        updates.append("interview_date = ?")
        params.append(date)
    if outcome:
        updates.append("outcome = ?")
        params.append(outcome)
    if notes:
        updates.append("notes = ?")
        params.append(notes)

    params.append(app_id)
    conn.execute(f"UPDATE applications SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    print(f"Application {app_id} updated: interview_status={interview_status}" +
          (f", date={date}" if date else "") +
          (f", outcome={outcome}" if outcome else "") +
          (f", notes={notes[:40]}" if notes else ""))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH}. Run state_machine.py first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Job application tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detail",  type=int,   metavar="ID",     help="Show full details for application ID")
    parser.add_argument("--filter",  type=str,   metavar="STATUS", help="Filter by interview status (or 'interview' for all non-none)")
    parser.add_argument("--stats",   action="store_true",          help="Show statistics and skill analysis")
    parser.add_argument("--update",  type=int,   metavar="ID",     help="Application ID to update")
    parser.add_argument("status",    nargs="?",  default=None,     help="New interview status (used with --update)")
    parser.add_argument("--date",    type=str,   metavar="DATE",   help="Interview date (YYYY-MM-DD)")
    parser.add_argument("--outcome", type=str,                     help="Outcome: rejected | offer | withdrawn")
    parser.add_argument("--notes",   type=str,                     help="Free-text notes")

    args = parser.parse_args()

    if args.stats:
        cmd_stats()
    elif args.detail:
        cmd_detail(args.detail)
    elif args.update:
        if not args.status:
            parser.error("--update requires a status argument: python3 db_viewer.py --update <ID> <status>")
        cmd_update(args.update, args.status, date=args.date, outcome=args.outcome, notes=args.notes)
    else:
        cmd_list(filter_status=args.filter)


if __name__ == "__main__":
    main()
