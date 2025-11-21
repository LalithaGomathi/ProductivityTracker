# Agent Productivity Analyzer — GenAI Starter

This is a starter Streamlit app that demonstrates a GenAI-friendly solution for agent productivity analysis.

## What it does
- Accepts CSV uploads for tickets and calls (and optional schedule CSV).
- Computes:
  - Productive time per agent (hours/minutes)
  - Scheduled work time for the same period (from schedule or default shift hours)
  - Utilization percentage
  - Average handling time by category
  - Idle/unaccounted time
- Dashboard includes KPI cards, per-agent table, category averages, heatmap, and CSV export.

## How to run (locally)
1. Create a Python 3.10+ virtualenv.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Open http://localhost:8501

## Sample CSV formats
- `tickets.csv`: agent_id,ticket_id,category,start_ts,end_ts
- `calls.csv`: agent_id,call_id,category,start_ts,end_ts,duration_seconds
- `schedule.csv`: agent_id,date,shift_start,shift_end

## Notes & next steps
- Add connectors for ServiceNow (API) and AWS Connect (S3 or API) to pull records directly.
- Replace simple CSV mapping with a persistent category mapping UI and save preferences.
- Add auth, RBAC, and data retention controls for production.
- Consider moving heavy processing to a background worker (e.g., Celery) for large datasets.