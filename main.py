"""
FasalSetu — FastAPI application.
All query endpoints are sync (def, not async def) because Runner.run()
is a sync generator that manages its own asyncio thread internally.
"""
import logging
import os
import shutil
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.orchestrator import run_query
from compliance.guardrail import get_audit_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fasalsetu")

app = FastAPI(title="FasalSetu Agricultural AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


class QueryRequest(BaseModel):
    query:      str
    location:   str = ""
    crop:       str = ""
    soil_data:  dict = {}
    session_id: str = "default"


class PesticideCheckRequest(BaseModel):
    pesticide_name: str


@app.get("/")
def root():
    index = os.path.join(_static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "FasalSetu API running. See /docs"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "agents":    ["soil", "disease", "market", "weather"],
        "model":     "gemini-2.5-flash",
        "framework": "google-adk",
    }


@app.post("/query")
def query(req: QueryRequest):
    """Main orchestrator — sync def required, do not change to async def."""
    try:
        context = {k: v for k, v in {
            "location":  req.location,
            "crop":      req.crop,
            "soil_data": req.soil_data,
        }.items() if v}
        return run_query(req.query, context=context or None, session_id=req.session_id)
    except Exception as e:
        logger.error("Query error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": traceback.format_exc()})


@app.post("/analyze-image")
def analyze_image(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return run_query(
            "Analyze this crop image for disease and recommend treatment.",
            context={"image_path": tmp_path},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/check-pesticide")
def check_pesticide(req: PesticideCheckRequest):
    """Compliance demo — try {"pesticide_name": "Endosulfan"}"""
    return run_query(f"Is {req.pesticide_name} safe to use on crops in India?")


@app.get("/audit-log")
def audit_log(last_n: int = 20):
    log = get_audit_log(last_n)
    return {"total": len(log), "entries": log}


@app.get("/audit")
def audit(last_n: int = 20):
    return audit_log(last_n)
