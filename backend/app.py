from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
from datetime import datetime
from agents.designer_agent import DesignerAgent
from agents.report_writer_agent import ReportWriterAgent
from agents.summarizer_agent import SummarizerAgent
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

agent = DesignerAgent()
report_writer = ReportWriterAgent()
summarizer = SummarizerAgent(gemini_api_key=GEMINI_API_KEY)

app = FastAPI()


# ========== UPDATED MODELS (task_type removed) ==========
class RequestModel(BaseModel):
    topic: str
    # task_type removed - no longer needed


class SummarizeModel(BaseModel):
    topic: str


class ReportRequest(BaseModel):
    topic: str
    format: str = "markdown"


# ========== ENDPOINTS ==========

@app.post("/summarize")
async def summarize(req: SummarizeModel):
    result = await summarizer.run(req.dict())
    return result


@app.post("/run")
async def run_agent(req: RequestModel):
    # Pass only topic, task_type removed
    result = await agent.run({"topic": req.topic})
    return result


@app.post("/write_report")
async def write_report(req: ReportRequest):
    result = await report_writer.run(req.dict())
    return result


@app.post("/download")
async def download_report(req: ReportRequest):
    result = await report_writer.run({
        "topic": req.topic,
        "format": "markdown"
    })
    
    content = result['content']
    filename = result['filename']
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return FileResponse(
        filepath,
        media_type="text/markdown",
        filename=filename
    )


@app.post("/download-docx")
async def download_docx(req: ReportRequest):
    import pypandoc
    import uuid
    
    result = await report_writer.run({
        "topic": req.topic,
        "format": "latex"
    })
    
    tex_content = result["content"]
    
    # Create temp directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    tex_file = os.path.join(temp_dir, f"{uuid.uuid4()}.tex")
    docx_file = os.path.join(temp_dir, f"{uuid.uuid4()}.docx")

    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)

    pypandoc.convert_file(
        tex_file,
        "docx",
        outputfile=docx_file,
        extra_args=[]
    )

    return FileResponse(
        docx_file,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{req.topic.replace(' ', '_')}.docx"
    )


@app.post("/download-tex")
async def download_tex(req: ReportRequest):
    """Download raw LaTeX file"""
    result = await report_writer.run({
        "topic": req.topic,
        "format": "latex"
    })

    if result.get("status") != "ok":
        return result

    tex_content = result["content"]
    filename = result.get("filename", f"{req.topic.replace(' ', '_')}.tex")
    safe_filename = filename.replace("/", "_")

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", safe_filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tex_content)

    return FileResponse(
        filepath,
        media_type="application/x-tex",
        filename=safe_filename
    )


@app.post("/download-overleaf")
async def download_overleaf(req: ReportRequest):
    """Download Overleaf-ready ZIP with LaTeX + IEEEtran.cls"""
    import zipfile
    import tempfile
    import requests

    # 1. Generate LaTeX report
    try:
        result = await report_writer.run({
            "topic": req.topic,
            "format": "latex"
        })
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if "content" not in result:
        return {
            "status": "error",
            "message": "Report writer did not return LaTeX content."
        }

    tex_content = result["content"]

    # 2. Create temp directory
    temp_dir = tempfile.mkdtemp()
    main_tex_path = os.path.join(temp_dir, "main.tex")

    with open(main_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # 3. Download IEEEtran.cls (correct class file)
    ieee_cls_path = os.path.join(temp_dir, "IEEEtran.cls")
    cls_url = "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls"

    try:
        cls_response = requests.get(cls_url, timeout=10)
        cls_response.raise_for_status()
        
        with open(ieee_cls_path, "wb") as f:
            f.write(cls_response.content)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to download IEEEtran.cls: {str(e)}"
        }

    # 4. Create empty refs.bib
    refs_path = os.path.join(temp_dir, "refs.bib")
    with open(refs_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated bibliography file\n")

    # 5. Create ZIP (files in root, not nested)
    zip_filename = f"{req.topic.replace(' ', '_')}_overleaf.zip"
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    zip_path = os.path.join("outputs", zip_filename)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Use arcname to put files in ZIP root
        zipf.write(main_tex_path, arcname="main.tex")
        zipf.write(ieee_cls_path, arcname="IEEEtran.cls")
        zipf.write(refs_path, arcname="refs.bib")

    # 6. Return ZIP file
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_filename
    )
from pipeline import router as pipeline_router
app.include_router(pipeline_router)
