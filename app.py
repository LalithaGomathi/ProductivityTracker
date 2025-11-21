
import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import requests

st.set_page_config(page_title="Agent Productivity - Deluxe", layout="wide")
st.title("Agent Productivity — Deluxe Streamlit UI")

st.sidebar.header("Data Inputs")
tickets_file = st.sidebar.file_uploader("Tickets CSV", type=["csv"])
calls_file = st.sidebar.file_uploader("Calls CSV", type=["csv"])
schedule_file = st.sidebar.file_uploader("Schedule CSV", type=["csv"])

st.sidebar.markdown("Or use backend connectors (ServiceNow / AWS Connect)")
use_backend = st.sidebar.checkbox("Use backend API for connectors", value=False)
if use_backend:
    api_url = st.sidebar.text_input("Backend API URL", value="http://backend:8000/upload/records")

if st.button("Upload & Analyze (local)"):
    files = {}
    if tickets_file: files['tickets'] = tickets_file.getvalue().decode('utf-8')
    if calls_file: files['calls'] = calls_file.getvalue().decode('utf-8')
    if schedule_file: files['schedule'] = schedule_file.getvalue().decode('utf-8')
    # call local analysis endpoint if backend selected
    if use_backend:
        m = {}
        fd = {}
        if tickets_file: fd['tickets'] = ('tickets.csv', tickets_file, 'text/csv')
        if calls_file: fd['calls'] = ('calls.csv', calls_file, 'text/csv')
        if schedule_file: fd['schedule'] = ('schedule.csv', schedule_file, 'text/csv')
        try:
            resp = requests.post(f"{api_url}", files=fd, timeout=30)
            st.success(f"Uploaded: {resp.status_code} - {resp.text}")
            resp2 = requests.post(f"{api_url}/analyze", data={'default_shift_hours':8.0,'overlap_rule':'split'}, timeout=60)
            st.json(resp2.json())
        except Exception as e:
            st.error(f"Backend request failed: {e}")
    else:
        st.info("Running local client-side analysis (no backend)")
        # minimal display: show uploaded CSVs
        if tickets_file:
            df_t = pd.read_csv(StringIO(tickets_file.getvalue().decode('utf-8')))
            st.subheader("Tickets sample")
            st.dataframe(df_t.head())
        if calls_file:
            df_c = pd.read_csv(StringIO(calls_file.getvalue().decode('utf-8')))
            st.subheader("Calls sample")
            st.dataframe(df_c.head())

st.markdown("---")
st.markdown("This Streamlit app is a lightweight UI. For production-grade features, use the backend service included in the package.")
