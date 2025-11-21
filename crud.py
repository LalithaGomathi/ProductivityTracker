
import os, pandas as pd
from io import StringIO, BytesIO
BASE = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(BASE, exist_ok=True)

def save_csv_bytes(name, bts):
    path = os.path.join(BASE, name)
    with open(path, 'wb') as f:
        if isinstance(bts, str): bts = bts.encode('utf-8')
        f.write(bts)

def load_csv_to_df(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path): return None
    try:
        return pd.read_csv(path)
    except Exception:
        with open(path, 'rb') as f:
            return pd.read_csv(StringIO(f.read().decode('utf-8')))

def ensure_ts(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def intervals_from_rows(df):
    if df is None: return []
    rows = []
    for _, r in df.iterrows():
        s = None; e = None
        for c in df.columns:
            if 'start' in c.lower():
                s = r[c]; break
        for c in df.columns:
            if 'end' in c.lower():
                e = r[c]; break
        if pd.isna(s) or pd.isna(e): continue
        rows.append((r.get('agent_id'), pd.to_datetime(s), pd.to_datetime(e), r.get('category')))
    return rows

def sum_durations_union(intervals):
    if not intervals: return 0.0
    ints = sorted(intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = ints[0]
    for s,e in ints[1:]:
        if s <= cur_e: cur_e = max(cur_e, e)
        else:
            merged.append((cur_s,cur_e)); cur_s,cur_e = s,e
    merged.append((cur_s,cur_e))
    return sum((e-s).total_seconds() for s,e in merged)

def compute_metrics(tickets_df, calls_df, schedule_df, default_shift_hours=8.0, overlap_rule='split'):
    if tickets_df is not None:
        tickets_df = ensure_ts(tickets_df, ['start_ts','end_ts'])
    if calls_df is not None:
        calls_df = ensure_ts(calls_df, ['start_ts','end_ts'])
    ticket_rows = intervals_from_rows(tickets_df) if tickets_df is not None else []
    call_rows = intervals_from_rows(calls_df) if calls_df is not None else []
    per_agent = {}
    per_agent_cat = {}
    agents = set()
    for ag,s,e,cat in ticket_rows + call_rows:
        agents.add(ag)
        per_agent.setdefault(ag, []).append((s,e))
        per_agent_cat.setdefault((ag, cat), []).append((s,e))
    results = []
    for ag in sorted(agents):
        ints = per_agent.get(ag, [])
        if overlap_rule=='split':
            productive = sum_durations_union(ints)
        else:
            productive = sum((e-s).total_seconds() for s,e in ints)
        # scheduled seconds (simple)
        scheduled = 0.0
        if schedule_df is not None:
            sd = schedule_df[schedule_df.get('agent_id')==ag] if 'agent_id' in schedule_df.columns else schedule_df
            for _, row in sd.iterrows():
                try:
                    d = pd.to_datetime(row['date']).date()
                    st = pd.to_datetime(str(d) + ' ' + str(row.get('shift_start')))
                    et = pd.to_datetime(str(d) + ' ' + str(row.get('shift_end')))
                    scheduled += max((et-st).total_seconds(), 0)
                except Exception:
                    scheduled += default_shift_hours*3600
        else:
            days = set((s.date() for s,e in ints))
            scheduled = len(days) * default_shift_hours*3600
        util = productive/scheduled*100.0 if scheduled>0 else None
        results.append({'agent_id':ag,'productive_seconds':productive,'scheduled_seconds':scheduled,'utilization_pct':util})
    return {'per_agent': results}
