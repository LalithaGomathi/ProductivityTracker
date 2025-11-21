
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
import zipfile
import io

st.set_page_config(page_title="Agent Productivity — GenAI Starter", layout="wide")

st.title("Agent Productivity Analyzer — GenAI Starter")
st.write("Upload service-desk ticket exports and phone-call exports (CSV). Or optionally upload a schedule CSV. "
         "This demo computes productive time per agent, scheduled work time (if provided or using defaults), utilization, "
         "average handling time by category, idle/unaccounted time, and provides a heatmap and CSV export.")

with st.expander("Expected sample CSV formats (examples included in repo) ✅", expanded=False):
    st.markdown("""
    **tickets.csv** (required/optional depending on input)
    - agent_id, ticket_id, category, start_ts, end_ts
    - `start_ts` and `end_ts` should be ISO datetimes (e.g. 2025-11-20 09:12:00)
    
    **calls.csv**
    - agent_id, call_id, category, start_ts, end_ts, duration_seconds (duration optional)
    
    **schedule.csv** (optional)
    - agent_id, date, shift_start, shift_end  (times as HH:MM or ISO)
    """)

# Upload area
st.sidebar.header("Inputs")
tickets_file = st.sidebar.file_uploader("Upload tickets CSV", type=["csv"])
calls_file = st.sidebar.file_uploader("Upload calls CSV", type=["csv"])
schedule_file = st.sidebar.file_uploader("Upload schedule CSV (optional)", type=["csv"])

st.sidebar.header("Settings")
default_shift_hours = st.sidebar.number_input("Default shift hours (per day)", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
overlap_handling = st.sidebar.selectbox("Overlapping tasks rule", options=["Split overlapping time (union)","Count each task fully (sum)"], index=0)
category_map_text = st.sidebar.text_area("Category mapping (old->new lines)", value="") 
# parse category mappings
category_map = {}
for line in category_map_text.splitlines():
    if "->" in line:
        old,new = line.split("->",1)
        category_map[old.strip()] = new.strip()

def read_csv_uploader(f):
    try:
        return pd.read_csv(f)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def ensure_ts(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def map_category(cat):
    return category_map.get(cat, cat)

def parse_and_normalize_tickets(df):
    df = df.copy()
    expected = ['agent_id','ticket_id','category','start_ts','end_ts']
    # Allow alternate names
    rename_map = {}
    for col in df.columns:
        l = col.lower()
        if l in expected:
            rename_map[col] = l
        elif "agent" in l and "id" in l:
            rename_map[col] = 'agent_id'
        elif "ticket" in l and ("id" in l or "number" in l):
            rename_map[col] = 'ticket_id'
        elif "category" in l or "type" in l:
            rename_map[col] = 'category'
        elif "start" in l and ("time" in l or "ts" in l or "date" in l):
            if 'start_ts' not in rename_map.values():
                rename_map[col] = 'start_ts'
        elif "end" in l and ("time" in l or "ts" in l or "date" in l or "closed" in l):
            if 'end_ts' not in rename_map.values():
                rename_map[col] = 'end_ts'
    df = df.rename(columns=rename_map)
    df = ensure_ts(df, ['start_ts','end_ts'])
    if 'end_ts' not in df or df['end_ts'].isna().all():
        st.warning("No end timestamps found for tickets — rows with missing end will be ignored for duration calculations.")
    df['category'] = df.get('category','').fillna('').astype(str).apply(map_category)
    return df

def parse_and_normalize_calls(df):
    df = df.copy()
    expected = ['agent_id','call_id','category','start_ts','end_ts','duration_seconds']
    rename_map = {}
    for col in df.columns:
        l = col.lower()
        if l in expected:
            rename_map[col] = l
        elif "agent" in l and "id" in l:
            rename_map[col] = 'agent_id'
        elif "call" in l and ("id" in l or "number" in l):
            rename_map[col] = 'call_id'
        elif "category" in l or "type" in l:
            rename_map[col] = 'category'
        elif "start" in l and ("time" in l or "ts" in l or "date" in l):
            if 'start_ts' not in rename_map.values():
                rename_map[col] = 'start_ts'
        elif "end" in l and ("time" in l or "ts" in l or "date" in l):
            if 'end_ts' not in rename_map.values():
                rename_map[col] = 'end_ts'
        elif "duration" in l:
            rename_map[col] = 'duration_seconds'
    df = df.rename(columns=rename_map)
    df = ensure_ts(df, ['start_ts','end_ts'])
    if 'duration_seconds' not in df.columns:
        # compute duration
        if 'start_ts' in df.columns and 'end_ts' in df.columns:
            df['duration_seconds'] = (df['end_ts'] - df['start_ts']).dt.total_seconds().fillna(0)
    df['category'] = df.get('category','').fillna('').astype(str).apply(map_category)
    return df

def intervals_from_rows(df):
    # expects start_ts and end_ts and agent_id
    rows = []
    for _, r in df.dropna(subset=['start_ts']).iterrows():
        start = r['start_ts']
        end = r['end_ts'] if pd.notna(r.get('end_ts')) else (start + pd.Timedelta(seconds=r.get('duration_seconds',0)))
        if pd.isna(end):
            continue
        if end < start:
            # swap or skip
            end = start
        rows.append((r['agent_id'], start, end, r.get('category')))
    return rows

def sum_durations_union(intervals):
    # intervals: list of (start,end)
    if not intervals:
        return 0.0
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = intervals_sorted[0]
    for s,e in intervals_sorted[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s,cur_e))
            cur_s,cur_e = s,e
    merged.append((cur_s,cur_e))
    total = sum((e-s).total_seconds() for s,e in merged)
    return total

def compute_metrics(tickets_df, calls_df, schedule_df=None):
    # normalize
    ticket_rows = intervals_from_rows(tickets_df) if tickets_df is not None else []
    call_rows = intervals_from_rows(calls_df) if calls_df is not None else []
    # per-agent collection
    per_agent_intervals = {}
    per_agent_category = {}
    all_agents = set()
    for ag, s,e,cat in ticket_rows + call_rows:
        all_agents.add(ag)
        per_agent_intervals.setdefault(ag, []).append((s,e))
        per_agent_category.setdefault((ag, cat), []).append((s,e))
    # compute productive time per agent
    results = []
    for ag in sorted(all_agents):
        intervals = per_agent_intervals.get(ag, [])
        if overlap_handling.startswith("Split"):
            productive_seconds = sum_durations_union(intervals)
        else:
            productive_seconds = sum((e-s).total_seconds() for s,e in intervals)
        # find date range for scheduled time: use schedule if provided; else derive unique dates in intervals
        scheduled_seconds = 0.0
        if schedule_df is not None and not schedule_df.empty:
            sd = schedule_df[schedule_df['agent_id']==ag]
            # expect date, shift_start, shift_end
            for _, row in sd.iterrows():
                try:
                    d = pd.to_datetime(row['date']).date()
                    ss = row.get('shift_start')
                    se = row.get('shift_end')
                    if pd.isna(ss) or pd.isna(se):
                        scheduled_seconds += default_shift_hours*3600.0
                    else:
                        # parse times
                        fmt = "%H:%M"
                        def parse_time(val):
                            if isinstance(val, str) and ":" in val:
                                t = datetime.strptime(val.strip(), "%H:%M").time()
                                return t
                            elif isinstance(val, pd.Timestamp):
                                return val.time()
                            return None
                        t1 = parse_time(ss)
                        t2 = parse_time(se)
                        if t1 and t2:
                            dt = datetime.combine(d,t2) - datetime.combine(d,t1)
                            scheduled_seconds += max(dt.total_seconds(), 0)
                        else:
                            scheduled_seconds += default_shift_hours*3600.0
                except Exception as e:
                    scheduled_seconds += default_shift_hours*3600.0
        else:
            # derive unique days from intervals
            days = set((s.date() for s,e in intervals))
            scheduled_seconds = len(days) * (default_shift_hours*3600.0)
        utilization = (productive_seconds / scheduled_seconds * 100.0) if scheduled_seconds>0 else None
        results.append({
            'agent_id': ag,
            'productive_seconds': productive_seconds,
            'scheduled_seconds': scheduled_seconds,
            'utilization_pct': utilization
        })
    df_results = pd.DataFrame(results)
    # average handling time by category across agents
    cat_list = []
    for (ag, cat), ints in per_agent_category.items():
        total = 0.0
        for s,e in ints:
            total += (e-s).total_seconds()
        count = len(ints)
        cat_list.append({'agent_id': ag, 'category': cat, 'total_seconds': total, 'count': count, 'avg_seconds': total/count if count>0 else 0})
    df_cat = pd.DataFrame(cat_list)
    # idle/unaccounted time per agent = scheduled - productive (if positive)
    if not df_results.empty:
        df_results['idle_seconds'] = df_results['scheduled_seconds'] - df_results['productive_seconds']
        df_results['idle_seconds'] = df_results['idle_seconds'].apply(lambda x: max(x,0.0))
    # heatmap data (day of week x hour) aggregation of productive seconds
    heat = {}
    for ag, s,e,cat in ticket_rows + call_rows:
        cur = s
        while cur < e:
            next_hour = (cur + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            slice_end = min(next_hour, e)
            hour_key = (cur.weekday(), cur.hour)  # weekday 0=Mon
            heat[hour_key] = heat.get(hour_key, 0.0) + (slice_end-cur).total_seconds()
            cur = slice_end
    heat_df = pd.DataFrame([{'weekday':k[0],'hour':k[1],'seconds':v} for k,v in heat.items()])
    return df_results, df_cat, heat_df

# Read uploads
tickets_df = read_csv_uploader(tickets_file) if tickets_file else None
calls_df = read_csv_uploader(calls_file) if calls_file else None
schedule_df = read_csv_uploader(schedule_file) if schedule_file else None

if tickets_df is not None:
    tickets_df = parse_and_normalize_tickets(tickets_df)
if calls_df is not None:
    calls_df = parse_and_normalize_calls(calls_df)
if schedule_df is not None:
    schedule_df = schedule_df.rename(columns={c:c.lower() for c in schedule_df.columns})

if st.button("Run analysis"):
    if tickets_df is None and calls_df is None:
        st.error("Please upload at least one of tickets.csv or calls.csv to analyze.")
    else:
        df_results, df_cat, heat_df = compute_metrics(tickets_df, calls_df, schedule_df)
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        total_prod = df_results['productive_seconds'].sum() if not df_results.empty else 0
        total_sched = df_results['scheduled_seconds'].sum() if not df_results.empty else 0
        avg_util = df_results['utilization_pct'].mean() if not df_results.empty else None
        col1.metric("Total productive time", f\"{int(total_prod//3600)}h {int((total_prod%3600)//60)}m\")
        col2.metric("Total scheduled time", f\"{int(total_sched//3600)}h {int((total_sched%3600)//60)}m\")
        col3.metric("Average utilization", f\"{avg_util:.1f}%\" if avg_util is not None else \"N/A\")
        col4.metric("Agents analyzed", len(df_results))

        st.markdown(\"---\")
        st.subheader(\"Per-agent table\")
        if not df_results.empty:
            df_display = df_results.copy()
            df_display['productive_hm'] = df_display['productive_seconds'].apply(lambda s: f\"{int(s//3600)}h {int((s%3600)//60)}m\")
            df_display['scheduled_hm'] = df_display['scheduled_seconds'].apply(lambda s: f\"{int(s//3600)}h {int((s%3600)//60)}m\")
            df_display['idle_hm'] = df_display['idle_seconds'].apply(lambda s: f\"{int(s//3600)}h {int((s%3600)//60)}m\")
            df_display['util_pct'] = df_display['utilization_pct'].apply(lambda x: f\"{x:.1f}%\" if pd.notna(x) else 'N/A')
            st.dataframe(df_display[['agent_id','productive_hm','scheduled_hm','util_pct','idle_hm']])

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Export results CSV", data=csv, file_name="agent_productivity_results.csv", mime='text/csv')

        st.subheader(\"Average handling time by category\")
        if not df_cat.empty:
            agg = df_cat.groupby('category').agg({'total_seconds':'sum','count':'sum'}).reset_index()
            agg['avg_seconds'] = agg['total_seconds']/agg['count']
            agg['avg_hm'] = agg['avg_seconds'].apply(lambda s: f\"{int(s//60)}m {int(s%60)}s\")
            st.dataframe(agg[['category','count','avg_hm']])
        else:
            st.write(\"No category data available.\")

        st.subheader(\"Heatmap: productived seconds by weekday (0=Mon) and hour\")
        if not heat_df.empty:
            # pivot
            pivot = heat_df.pivot_table(index='weekday', columns='hour', values='seconds', aggfunc='sum', fill_value=0)
            fig, ax = plt.subplots(figsize=(12,3))
            im = ax.imshow(pivot, aspect='auto')
            ax.set_yticks(range(7)); ax.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:pivot.shape[0]])
            ax.set_xlabel('Hour of day'); ax.set_ylabel('Weekday')
            ax.set_title('Productive seconds (heat)')
            st.pyplot(fig)
        else:
            st.write(\"No heatmap data available.\")

        st.success(\"Analysis complete.\")


st.markdown(\"---\")
st.caption(\"This sample app is a starter. For production, integrate secure authentication, scalable storage, direct connectors to ServiceNow/AWS Connect, and formal testing.\")
