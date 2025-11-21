
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd, os
from . import crud
app = FastAPI(title="Agent Productivity Backend")

@app.get('/')
def root():
    return {'status':'ok','message':'Agent Productivity Backend'}

@app.post('/upload/records')
async def upload_records(tickets: UploadFile = File(None), calls: UploadFile = File(None), schedule: UploadFile = File(None)):
    saved = {}
    if tickets:
        content = await tickets.read()
        crud.save_csv_bytes('tickets.csv', content)
        saved['tickets'] = 'saved'
    if calls:
        content = await calls.read()
        crud.save_csv_bytes('calls.csv', content)
        saved['calls'] = 'saved'
    if schedule:
        content = await schedule.read()
        crud.save_csv_bytes('schedule.csv', content)
        saved['schedule'] = 'saved'
    if not saved:
        raise HTTPException(status_code=400, detail='No files uploaded')
    return JSONResponse({'status':'ok','saved':saved})

@app.post('/analyze')
async def analyze(default_shift_hours: float = Form(8.0), overlap_rule: str = Form('split')):
    df_t = crud.load_csv_to_df('tickets.csv')
    df_c = crud.load_csv_to_df('calls.csv')
    df_s = crud.load_csv_to_df('schedule.csv')
    if df_t is None and df_c is None:
        raise HTTPException(status_code=400, detail='No data available')
    result = crud.compute_metrics(df_t, df_c, df_s, default_shift_hours, overlap_rule)
    return JSONResponse(result)

@app.get('/download/samples')
def download_samples():
    return FileResponse(os.path.join(os.path.dirname(__file__), '..', 'samples', 'sample_input_files.zip'), media_type='application/zip', filename='sample_input_files.zip')
