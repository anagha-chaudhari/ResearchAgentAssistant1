from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os
from datetime import datetime
from agents.designer_agent import DesignerAgent
from agents.report_writer_agent import ReportWriterAgent
from agents.summarizer_agent import SummarizerAgent

agent = DesignerAgent()
report_writer = ReportWriterAgent()
summarizer = SummarizerAgent()

app = FastAPI()

class RequestModel(BaseModel):
    topic : str
    task_type : str = "thermal_simulation"
class SummarizeModel(BaseModel):
    topic: str
class ReportRequest(BaseModel):
    topic: str
    format: str = "markdown"
  
@app.post("/summarize")
async def summarize(req:SummarizeModel):
    result=await summarizer.run(req.dict())
    return result

@app.post("/run")
async def run_agent(req:RequestModel):
    result = await agent.run(req.dict())
    return result

@app.post("/write_report")
async def write_report(req: ReportRequest):
    result=await report_writer.run(req.dict())
    return result

@app.post("/download")
async def download_report(req:ReportRequest):
    result = await report_writer.run({
        "topic":req.topic,
        "format":"markdown"
    })
    
    content=result['content']
    filename=result['filename']
    filepath=f'./{filename}'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return FileResponse(filepath,media_type="text/markdown",filename=filename)

@app.post("/download-docx")
async def download_docx(req:ReportRequest):
    import pypandoc
    import uuid
    
    result=await report_writer.run({
        "topic":req.topic,
        "format":"latex"
    })
    
    tex_content = result["content"]
    tex_file = f"{uuid.uuid4()}.tex"
    docx_file = f"{uuid.uuid4()}.docx"

    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)

    pypandoc.convert_file(
        tex_file,
        "docx",
        outputfile=docx_file,
        extra_args=[]
    )

    return FileResponse(docx_file, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=f"{req.topic}.docx")
@app.post("/download-tex")
async def download_tex(req: ReportRequest):
    # Always use IEEE LaTeX for Overleaf
    result = await report_writer.run({
        "topic": req.topic,
        "format": "latex"
    })

    if result.get("status") != "ok":
        return result

    tex_content = result["content"]
    filename = result.get("filename", f"{req.topic.replace(' ', '_')}.tex")
    safe_filename = filename.replace("/", "_")

    filepath = os.path.abspath(safe_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tex_content)

    return FileResponse(
        filepath,
        media_type="application/x-tex",
        filename=safe_filename
    )
    
@app.post("/download-overleaf")
async def download_overleaf(req: ReportRequest):
    import os
    import zipfile
    import uuid
    import tempfile
    import requests

    # 1. Run Report Writer Agent with IEEE LaTeX mode
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

    # 2. Create temp directory for Overleaf project
    temp_dir = tempfile.mkdtemp()
    main_tex_path = os.path.join(temp_dir, "main.tex")

    # Write main.tex
    with open(main_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # 3. Download ieeeconf.cls (CTAN)
    ieeeconf_path = os.path.join(temp_dir, "ieeeconf.cls")
    cls_url = "https://mirrors.ctan.org/macros/latex/contrib/ieeeconf/ieeeconf.cls"

    try:
        cls_file = requests.get(cls_url, timeout=10)
        with open(ieeeconf_path, "wb") as f:
            f.write(cls_file.content)
    except Exception as e:
        return {"status": "error", "message": f"Failed to download ieeeconf.cls: {str(e)}"}

    # 4. Add empty refs.bib for reference support
    refs_path = os.path.join(temp_dir, "refs.bib")
    with open(refs_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated bibliography file\n")

    # 5. Prepare ZIP output
    zip_filename = f"{req.topic.replace(' ', '_')}_overleaf.zip"
    zip_path = os.path.join(temp_dir, zip_filename)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(main_tex_path, "main.tex")
        zipf.write(ieeeconf_path, "ieeeconf.cls")
        zipf.write(refs_path, "refs.bib")

    # 6. Return ZIP file
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_filename
    )