=== productivity_tracker/requirements.txt ===
pandas==2.2.2
numpy==1.26.4
streamlit==1.38.0
pyyaml==6.0.2
altair==5.3.0
python-dateutil==2.9.0.post0

=== productivity_tracker/README.md ===
# Agent Productivity Tracker

What it does:
- Combines ticket activity and phone-call time to compute productive minutes/hours, scheduled time, utilization %, idle time, and average handling time by category.
- Provides a heatmap (day/hour) to visualize busy times.
- Offers per-agent filters, team filters, compare views, and CSV export.

Quick start:
1. Create and activate a Python virtual environment.
2. Install: `pip install -r requirements.txt`.
3. Run: `streamlit run app.py`.
4. Upload your CSVs or use `sample_data/` to demo instantly.

Data expectations:

Tickets CSV:
- Columns: agent, ticket_id, category, start_ts, end_ts, (optional) team
- Each row is one handling window; multiple rows per ticket_id allowed.

Calls CSV:
- Columns: agent, call_id, category, start_ts, end_ts, duration_seconds (optional), (optional) team

Schedule CSV (optional):
- Columns: agent, date (YYYY-MM-DD), shift_start (HH:MM), shift_end (HH:MM), (optional) team
- If omitted, default shift 09:00–18:00 is applied per agent per day.

Settings:
- Overlap rule:
  - split_time: Overlapping windows are split proportionally so total time never double-counts.
  - count_full: Overlapping windows are fully counted.
- Category mapping: edit `config/category_mapping.json` to unify labels across systems.

Outputs:
- Per-agent KPIs: Productive Time, Scheduled Time, Utilization %, Idle Time.
- Team heatmap: busy hours by day/hour.
- Table: Average handling time by category (minutes).
- Export: `exports/per_agent_report.csv` and in-app download.

Notes:
- Ensure timestamps have consistent timezone and formats.
- For large files, consider batching or pre-aggregation.

=== productivity_tracker/settings.py ===
from dataclasses import dataclass
from datetime import time

@dataclass
class DefaultSettings:
    default_shift_start: time = time(9, 0)
    default_shift_end: time = time(18, 0)
    overlap_rule: str = "split_time"
    timezone: str = "Asia/Kolkata"

    ticket_columns = {
        "agent": "agent",
        "ticket_id": "ticket_id",
        "category": "category",
        "start_ts": "start_ts",
        "end_ts": "end_ts"
    }

    call_columns = {
        "agent": "agent",
        "call_id": "call_id",
        "category": "category",
        "start_ts": "start_ts",
        "end_ts": "end_ts",
        "duration_seconds": "duration_seconds"
    }

    schedule_columns = {
        "agent": "agent",
        "date": "date",
        "shift_start": "shift_start",
        "shift_end": "shift_end"
    }

=== productivity_tracker/compute.py ===
import pandas as pd
import numpy as np
from datetime import time
import json
import yaml
from typing import Dict, Tuple

from settings import DefaultSettings

def load_category_mapping(path: str) -> Dict[str, list]:
    with open(path, "r") as f:
        return json.load(f)

def load_app_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_datetimes(df: pd.DataFrame, start_col: str, end_col: str, tz_name: str) -> pd.DataFrame:
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    # Localize naive timestamps
    if getattr(df[start_col].dt, "tz", None) is None:
        df[start_col] = df[start_col].dt.tz_localize(tz_name, ambiguous="NaT", nonexistent="NaT")
    if getattr(df[end_col].dt, "tz", None) is None:
        df[end_col] = df[end_col].dt.tz_localize(tz_name, ambiguous="NaT", nonexistent="NaT")
    return df

def normalize_calls(df_calls: pd.DataFrame, settings: DefaultSettings, tz_name: str) -> pd.DataFrame:
    df = df_calls.copy()
    df = parse_datetimes(df, settings.call_columns["start_ts"], settings.call_columns["end_ts"], tz_name)
    dur_col = settings.call_columns["duration_seconds"]
    if dur_col not in df.columns or df[dur_col].isna().any():
        df[dur_col] = (df[settings.call_columns["end_ts"]] - df[settings.call_columns["start_ts"]]).dt.total_seconds()
    return df

def normalize_tickets(df_tickets: pd.DataFrame, settings: DefaultSettings, tz_name: str) -> pd.DataFrame:
    df = df_tickets.copy()
    df = parse_datetimes(df, settings.ticket_columns["start_ts"], settings.ticket_columns["end_ts"], tz_name)
    df["duration_seconds"] = (df[settings.ticket_columns["end_ts"]] - df[settings.ticket_columns["start_ts"]]).dt.total_seconds()
    return df

def normalize_schedule(df_sched: pd.DataFrame, settings: DefaultSettings) -> pd.DataFrame:
    df = df_sched.copy()
    df[settings.schedule_columns["date"]] = pd.to_datetime(df[settings.schedule_columns["date"]]).dt.date
    return df

def build_default_schedule(agents: pd.Series, dates: pd.Series, settings: DefaultSettings, tz_name: str, team_series: pd.Series = None) -> pd.DataFrame:
    rows = []
    start_t = settings.default_shift_start
    end_t = settings.default_shift_end
    # Pick one team per agent if present
    team_map = {}
    if team_series is not None and agents is not None:
        for a in agents.unique():
            try:
                team_map[a] = team_series[agents == a].dropna().iloc[0]
            except Exception:
                team_map[a] = None
    for a in agents.unique():
        for d in pd.Series(dates.unique()).sort_values():
            rows.append({
                settings.schedule_columns["agent"]: a,
                settings.schedule_columns["date"]: d,
                settings.schedule_columns["shift_start"]: start_t.strftime("%H:%M"),
                settings.schedule_columns["shift_end"]: end_t.strftime("%H:%M"),
                "team": team_map.get(a, None)
            })
    return pd.DataFrame(rows)

def apply_category_mapping(df: pd.DataFrame, category_col: str, mapping: Dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    reverse_map = {}
    for canonical, variants in mapping.items():
        for v in variants:
            reverse_map[str(v).lower()] = canonical
    def map_fn(x):
        if pd.isna(x):
            return "Other"
        x_str = str(x).lower()
        if x_str in reverse_map:
            return reverse_map[x_str]
        return x if x in mapping.keys() else "Other"
    df["category_mapped"] = df[category_col].apply(map_fn)
    return df

def overlap_adjust(events: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    events columns: agent, start_ts, end_ts, duration_seconds, source, category_mapped
    rule: 'count_full' or 'split_time'
    """
    out = []
    for agent, group in events.groupby("agent"):
        g = group.sort_values("start_ts").reset_index(drop=True)
        if rule == "count_full":
            g["productive_seconds"] = g["duration_seconds"].clip(lower=0)
            out.append(g)
            continue
        # split_time: allocate overlapping segments evenly
        boundaries = sorted(set(list(g["start_ts"]) + list(g["end_ts"])))
        alloc = np.zeros(len(g))
        for i in range(len(boundaries)-1):
            seg_start = boundaries[i]
            seg_end = boundaries[i+1]
            seg_len = (seg_end - seg_start).total_seconds()
            active = g[(g["start_ts"] < seg_end) & (g["end_ts"] > seg_start)]
            if seg_len > 0 and len(active) > 0:
                share = seg_len / len(active)
                for idx in active.index:
                    alloc[idx] += share
        g["productive_seconds"] = alloc
        out.append(g)
    return pd.concat(out, ignore_index=True)

def compute_kpis(df_tickets: pd.DataFrame,
                 df_calls: pd.DataFrame,
                 df_schedule: pd.DataFrame,
                 settings: DefaultSettings,
                 tz_name: str,
                 team_field: str = "team") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_t = normalize_tickets(df_tickets, settings, tz_name)
    df_c = normalize_calls(df_calls, settings, tz_name)

    mapping = load_category_mapping("config/category_mapping.json")
    df_t = apply_category_mapping(df_t, settings.ticket_columns["category"], mapping)
    df_c = apply_category_mapping(df_c, settings.call_columns["category"], mapping)

    t_cols = settings.ticket_columns
    c_cols = settings.call_columns

    events_t = df_t.rename(columns={
        t_cols["agent"]: "agent",
        t_cols["start_ts"]: "start_ts",
        t_cols["end_ts"]: "end_ts"
    })
    events_t["source"] = "Ticket"
    events_t["team"] = df_t[team_field] if team_field in df_t.columns else None

    events_c = df_c.rename(columns={
        c_cols["agent"]: "agent",
        c_cols["start_ts"]: "start_ts",
        c_cols["end_ts"]: "end_ts"
    })
    events_c["source"] = "Call"
    events_c["team"] = df_c[team_field] if team_field in df_c.columns else None

    events = pd.concat([
        events_t[["agent","start_ts","end_ts","duration_seconds","category_mapped","source","team"]],
        events_c[["agent","start_ts","end_ts","duration_seconds","category_mapped","source","team"]]
    ], ignore_index=True).dropna(subset=["agent","start_ts","end_ts"])

    if df_schedule is None or df_schedule.empty:
        dates = events["start_ts"].dt.date
        agents = events["agent"]
        df_schedule = build_default_schedule(agents, dates, settings, tz_name, team_series=events["team"])
    df_s = normalize_schedule(df_schedule, settings)

    rows = []
    for _, r in df_s.iterrows():
        agent = r[settings.schedule_columns["agent"]]
        date = r[settings.schedule_columns["date"]]
        team = r["team"] if "team" in df_s.columns else None
        start_t = time.fromisoformat(str(r[settings.schedule_columns["shift_start"]]))
        end_t = time.fromisoformat(str(r[settings.schedule_columns["shift_end"]]))
        start_dt = pd.to_datetime(f"{date} {start_t.strftime('%H:%M')}").tz_localize(tz_name)
        end_dt = pd.to_datetime(f"{date} {end_t.strftime('%H:%M')}").tz_localize(tz_name)
        rows.append({"agent": agent, "date": date, "shift_start": start_dt, "shift_end": end_dt, "team": team})
    schedule = pd.DataFrame(rows)

    events["date"] = events["start_ts"].dt.date
    merged = events.merge(schedule, on=["agent","date"], how="left", suffixes=("","_sched"))
    merged["sched_start"] = merged["shift_start"]
    merged["sched_end"] = merged["shift_end"]

    clipped_start = np.maximum(merged["start_ts"].view("int64"), merged["sched_start"].view("int64"))
    clipped_end = np.minimum(merged["end_ts"].view("int64"), merged["sched_end"].view("int64"))
    merged["clipped_duration"] = ((clipped_end - clipped_start) / 1e9).clip(lower=0)

    clipped_events = merged[["agent","date","start_ts","end_ts","clipped_duration","category_mapped","source","team"]].copy()
    clipped_events = clipped_events.rename(columns={"clipped_duration": "duration_seconds"})
    adjusted = overlap_adjust(clipped_events, settings.overlap_rule)

    sched_seconds = schedule.copy()
    sched_seconds["scheduled_seconds"] = (sched_seconds["shift_end"] - sched_seconds["shift_start"]).dt.total_seconds()

    daily_prod = adjusted.groupby(["agent","date","team"], dropna=False)["productive_seconds"].sum().reset_index()
    daily = daily_prod.merge(sched_seconds[["agent","date","scheduled_seconds","team"]], on=["agent","date","team"], how="left)
    daily["idle_seconds"] = (daily["scheduled_seconds"] - daily["productive_seconds"]).clip(lower=0)
    daily["utilization_pct"] = np.where(daily["scheduled_seconds"] > 0,
                                        100 * daily["productive_seconds"] / daily["scheduled_seconds"],
                                        np.nan)

    cat_aht = adjusted.groupby(["category_mapped","source"], dropna=False)["productive_seconds"].mean().reset_index()
    cat_aht = cat_aht.rename(columns={"productive_seconds": "avg_handle_seconds"})

    adjusted["hour"] = adjusted["start_ts"].dt.hour
    heatmap = adjusted.groupby(["date","hour","team"], dropna=False)["productive_seconds"].sum().reset_index()

    return daily, cat_aht, heatmap

=== productivity_tracker/ui.py ===
import streamlit as st
import pandas as pd
from settings import DefaultSettings

def sidebar_settings(settings: DefaultSettings):
    st.sidebar.header("Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        default_start = st.time_input("Default shift start", value=settings.default_shift_start)
    with col2:
        default_end = st.time_input("Default shift end", value=settings.default_shift_end)
    overlap_rule = st.sidebar.selectbox("Overlap rule", ["split_time", "count_full"], index=0)
    tz_name = st.sidebar.text_input("Timezone", value=settings.timezone)
    st.sidebar.caption("If schedule upload is missing, these defaults apply per agent per day.")
    return default_start, default_end, overlap_rule, tz_name

def filters(df_daily: pd.DataFrame):
    st.subheader("Filters")
    agents = sorted(df_daily["agent"].dropna().unique().tolist())
    teams = sorted(df_daily["team"].dropna().unique().tolist())
    dates = sorted(df_daily["date"].dropna().unique().tolist())
    sel_agents = st.multiselect("Agents", agents, default=agents[:2] if len(agents) >= 2 else agents)
    sel_teams = st.multiselect("Teams", teams, default=teams)
    date_range = st.date_input("Date range", value=(min(dates) if dates else None, max(dates) if dates else None))
    return sel_agents, sel_teams, date_range

=== productivity_tracker/app.py ===
import streamlit as st
import pandas as pd
import altair as alt
import os
from settings import DefaultSettings
from compute import compute_kpis, load_app_config
from ui import sidebar_settings, filters

st.set_page_config(page_title="Agent Productivity Tracker", layout="wide")
st.title("Agent Productivity Tracker")

os.makedirs("exports", exist_ok=True)

settings = DefaultSettings()
app_cfg = load_app_config("config/app_config.yaml")

st.sidebar.header("Upload data")
tickets_file = st.sidebar.file_uploader("Ticket logs CSV", type=["csv"])
calls_file = st.sidebar.file_uploader("Call logs CSV", type=["csv"])
sched_file = st.sidebar.file_uploader("Agent schedule CSV (optional)", type=["csv"])

st.sidebar.markdown("---")
default_start, default_end, overlap_rule, tz_name = sidebar_settings(settings)
settings.default_shift_start = default_start
settings.default_shift_end = default_end
settings.overlap_rule = overlap_rule
settings.timezone = tz_name

def read_csv_or_sample(file, sample_path):
    if file is not None:
        return pd.read_csv(file)
    return pd.read_csv(sample_path)

df_tickets = read_csv_or_sample(tickets_file, "sample_data/tickets_sample.csv")
df_calls = read_csv_or_sample(calls_file, "sample_data/calls_sample.csv")
df_schedule = None
if sched_file is not None:
    df_schedule = pd.read_csv(sched_file)

team_field = "team"

daily, cat_aht, heatmap = compute_kpis(df_tickets, df_calls, df_schedule, settings, tz_name, team_field=team_field)

sel_agents, sel_teams, date_range = filters(daily)

def apply_filters(df):
    dfx = df.copy()
    if sel_agents:
        dfx = dfx[dfx["agent"].isin(sel_agents)]
    if sel_teams:
        dfx = dfx[dfx["team"].isin(sel_teams)]
    if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
        start_d, end_d = date_range
        dfx = dfx[(dfx["date"] >= start_d) & (dfx["date"] <= end_d)]
    return dfx

daily_f = apply_filters(daily)
heatmap_f = apply_filters(heatmap)

st.subheader("Per-agent summary")
kpi_cols = st.columns(4)
for i, agent in enumerate(sorted(daily_f["agent"].unique())):
    agent_df = daily_f[daily_f["agent"] == agent]
    prod = agent_df["productive_seconds"].sum()
    sched = agent_df["scheduled_seconds"].sum()
    idle = agent_df["idle_seconds"].sum()
    util = (100 * prod / sched) if sched > 0 else 0
    with kpi_cols[i % 4]:
        st.metric(label=f"{agent} • Productive time", value=f"{int(prod//3600)}h {int((prod%3600)//60)}m")
        st.metric(label=f"{agent} • Scheduled time", value=f"{int(sched//3600)}h {int((sched%3600)//60)}m")
        st.metric(label=f"{agent} • Utilization %", value=f"{util:.1f}%")
        st.metric(label=f"{agent} • Idle time", value=f"{int(idle//3600)}h {int((idle%3600)//60)}m")

st.subheader("Team view: busiest hours heatmap")
if not heatmap_f.empty:
    heatmap_f["date_str"] = heatmap_f["date"].astype(str)
    heat = alt.Chart(heatmap_f).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("date_str:O", title="Date"),
        color=alt.Color("productive_seconds:Q", title="Productive seconds", scale=alt.Scale(scheme="blues")),
        tooltip=["team","date_str","hour","productive_seconds"]
    ).properties(height=300)
    st.altair_chart(heat, use_container_width=True)
else:
    st.info("No data for selected filters.")

st.subheader("Average handling time by category")
cat_view = cat_aht.copy()
cat_view["avg_handle_minutes"] = (cat_view["avg_handle_seconds"] / 60).round(1)
st.dataframe(cat_view[["category_mapped","source","avg_handle_minutes"]], use_container_width=True)

st.subheader("Compare view")
comp_cols = st.columns(2)
with comp_cols[0]:
    st.caption("Compare agents")
    agents_all = sorted(daily["agent"].unique().tolist())
    a1 = st.selectbox("Agent A", agents_all, index=0 if agents_all else None)
    a2 = st.selectbox("Agent B", agents_all, index=1 if len(agents_all) > 1 else 0)
    df_a1 = daily[daily["agent"] == a1]
    df_a2 = daily[daily["agent"] == a2]
    prod_a1 = df_a1["productive_seconds"].sum(); sched_a1 = df_a1["scheduled_seconds"].sum()
    prod_a2 = df_a2["productive_seconds"].sum(); sched_a2 = df_a2["scheduled_seconds"].sum()
    util_a1 = 100 * prod_a1 / sched_a1 if sched_a1 > 0 else 0
    util_a2 = 100 * prod_a2 / sched_a2 if sched_a2 > 0 else 0
    st.write(f"{a1}: Util {util_a1:.1f}% • Prod {int(prod_a1//3600)}h {int((prod_a1%3600)//60)}m")
    st.write(f"{a2}: Util {util_a2:.1f}% • Prod {int(prod_a2//3600)}h {int((prod_a2%3600)//60)}m")

with comp_cols[1]:
    st.caption("Compare weeks")
    daily["iso_week"] = pd.to_datetime(daily["date"]).dt.isocalendar().week
    weeks = sorted(daily["iso_week"].unique().tolist())
    w1 = st.selectbox("Week 1", weeks, index=0 if weeks else None)
    w2 = st.selectbox("Week 2", weeks, index=1 if len(weeks) > 1 else 0)
    df_w1 = daily[daily["iso_week"] == w1]
    df_w2 = daily[daily["iso_week"] == w2]
    prod_w1 = df_w1["productive_seconds"].sum(); sched_w1 = df_w1["scheduled_seconds"].sum()
    prod_w2 = df_w2["productive_seconds"].sum(); sched_w2 = df_w2["scheduled_seconds"].sum()
    util_w1 = 100 * prod_w1 / sched_w1 if sched_w1 > 0 else 0
    util_w2 = 100 * prod_w2 / sched_w2 if sched_w2 > 0 else 0
    st.write(f"Week {w1}: Util {util_w1:.1f}%")
    st.write(f"Week {w2}: Util {util_w2:.1f}%")

st.subheader("Export per-agent CSV")
export_agents = st.multiselect("Select agents to export", sorted(daily["agent"].unique().tolist()), default=sorted(daily["agent"].unique().tolist()))
if st.button("Export CSV"):
    export_df = daily[daily["agent"].isin(export_agents)].copy()
    export_df["productive_minutes"] = (export_df["productive_seconds"] / 60).round(1)
    export_df["scheduled_minutes"] = (export_df["scheduled_seconds"] / 60).round(1)
    export_df["idle_minutes"] = (export_df["idle_seconds"] / 60).round(1)
    export_path = os.path.join("exports", "per_agent_report.csv")
    export_df[["agent","team","date","productive_minutes","scheduled_minutes","idle_minutes","utilization_pct"]].to_csv(export_path, index=False)
    st.success(f"Exported to {export_path}")
    st.download_button("Download here", data=export_df.to_csv(index=False), file_name="per_agent_report.csv", mime="text/csv")

with st.expander("Raw daily data"):
    st.dataframe(daily, use_container_width=True)
with st.expander("Raw heatmap data"):
    st.dataframe(heatmap, use_container_width=True)

=== productivity_tracker/config/app_config.yaml ===
timezone: Asia/Kolkata
heatmap_hour_bins: 24
team_field_name: team

=== productivity_tracker/config/category_mapping.json ===
{
  "Incidents": ["Incident", "INC", "Break/Fix"],
  "Requests": ["Request", "REQ", "Service Request"],
  "Change": ["Change", "CHG"],
  "Calls": ["Call", "Voice", "Inbound", "Outbound"],
  "Other": ["Task", "Misc"]
}

=== productivity_tracker/sample_data/tickets_sample.csv ===
agent,ticket_id,category,start_ts,end_ts,team
Anita,T-1001,Incident,2025-11-10 09:15:00,2025-11-10 09:45:00,Team A
Anita,T-1002,Request,2025-11-10 10:00:00,2025-11-10 10:25:00,Team A
Rahul,T-2001,Incident,2025-11-10 09:30:00,2025-11-10 10:10:00,Team B
Rahul,T-2002,Change,2025-11-10 11:00:00,2025-11-10 11:50:00,Team B
Anita,T-1003,Incident,2025-11-11 14:10:00,2025-11-11 14:30:00,Team A

=== productivity_tracker/sample_data/calls_sample.csv ===
agent,call_id,category,start_ts,end_ts,duration_seconds,team
Anita,C-501,Inbound,2025-11-10 09:50:00,2025-11-10 10:00:00,600,Team A
Rahul,C-601,Outbound,2025-11-10 10:15:00,2025-11-10 10:35:00,1200,Team B
Anita,C-502,Inbound,2025-11-11 15:00:00,2025-11-11 15:12:00,720,Team A

=== productivity_tracker/sample_data/agent_schedule_sample.csv ===
agent,date,shift_start,shift_end,team
Anita,2025-11-10,09:00,18:00,Team A
Anita,2025-11-11,09:00,18:00,Team A
Rahul,2025-11-10,10:00,19:00,Team B
Rahul,2025-11-11,10:00,19:00,Team B
