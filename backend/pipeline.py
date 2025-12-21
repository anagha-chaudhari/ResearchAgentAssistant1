import os
import time
from typing import Optional, List, Dict, Any
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel

from agents.retrieval_agent import RetrievalAgent
from agents.memory_agent import MemoryAgent          
from agents.summarizer_agent import SummarizerAgent  
from agents.evaluator_agent import EvaluatorAgent    
from agents.designer_agent import DesignerAgent      
from agents.report_writer_agent import ReportWriterAgent  
from tools import memory_store    
from tools.report_history import save_report            
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from agents.rag_agent import ResearchRAG
from fastapi import APIRouter

router = APIRouter()



report_writer=ReportWriterAgent()
PROCESS_PIPELINE={}
# ---------- ENV VARS ----------
from dotenv import load_dotenv
load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SEMANTIC_SCHOLAR_API_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing SEMANTIC_SCHOLAR_API_KEY or GEMINI_API_KEY in .env!")

class ReportRequest(BaseModel):
    topic: str



# ---------- MODELS ----------
class PipelineRequest(BaseModel):
    topic: str
    max_papers: int = 10
    years_back: int = 3


# ---------- SMALL HELPER: MAP ANALYSIS -> memory_store ----------
def _sync_analysis_to_memory_store(
    topic: str,
    papers: List[Dict[str, Any]],
    analysis: Dict[str, Any],
):
    """
    Take teammate's SummarizerAgent output and push it into your tools.memory_store
    so DesignerAgent + ReportWriterAgent can work unchanged.
    """
    per_paper = analysis.get("per_paper", [])
    collected_gaps = analysis.get("collected_gaps", [])
    collected_citations = analysis.get("collected_citations", [])

    summaries_payload = []

    for paper, pp in zip(papers, per_paper):
        summary_text = pp.get("summary", "")
        gaps = pp.get("gaps", [])
        methods = pp.get("methods", [])
        baselines = pp.get("baselines", [])

        # Build a citation object compatible with your report builder
        citation = {
            "title": paper.get("title", "Untitled"),
            "authors": paper.get("authors", []),
            "year": paper.get("publication_year", None),
            "doi": paper.get("doi", "N/A"),
            "journal": paper.get("journal", "N/A"),
            "url": paper.get("url", "N/A"),
        }

        summaries_payload.append(
            {
                "title": paper.get("title", "Untitled"),
                "summary": summary_text,
                "gaps": gaps,
                "methods": methods,
                "key_findings": baselines,   # best-effort mapping
                "limitations": [],           # can be improved later
                "citations": citation,
            }
        )

    # Save into your in-process memory
    memory_store.save_summaries(topic, summaries_payload)

    # If global gaps from analysis exist, overwrite / augment
    if collected_gaps:
        memory_store.save_gaps(topic, collected_gaps)

    # Build citation list if you want a separate citations bucket
    citations_bucket = []
    for p in papers:
        citations_bucket.append(
            {
                "title": p.get("title", "Untitled"),
                "authors": p.get("authors", []),
                "year": p.get("publication_year", None),
                "doi": p.get("doi", "N/A"),
                "journal": p.get("journal", "N/A"),
                "url": p.get("url", "N/A"),
            }
        )
    if citations_bucket:
        memory_store.save_citations(topic, citations_bucket)


# ---------- PIPELINE ENDPOINT ----------
@router.post("/run_pipeline")
async def run_pipeline(req: PipelineRequest):
    """
    Full pipeline:
    1. Retrieve papers (Semantic Scholar) + optional datasets (Google CSE)
    2. Store raw papers in JSON memory (research_memory.json)
    3. Summarize + aggregate gaps/methods/datasets (Gemini)
    4. Evaluate quality (Gemini)
    5. Store analysis + evaluation (research_analysis.json)
    6. Sync to tools.memory_store for downstream agents
    7. Run DesignerAgent -> experiment plan
    8. Run ReportWriterAgent -> markdown + IEEE LaTeX
    """

    # ---------- 1. Instantiate agents ----------
    retrieval = RetrievalAgent(
        semantic_api_key=SEMANTIC_SCHOLAR_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        max_results=req.max_papers,
    )

    # memory of record = backend/data/...
    memory_dir = os.path.join("backend", "data")
    os.makedirs(memory_dir, exist_ok=True)

    memory_file = os.path.join(memory_dir, "research_memory.json")
    analysis_file = os.path.join(memory_dir, "research_analysis.json")

    json_memory = MemoryAgent(
        memory_file=memory_file,
        analysis_file=analysis_file,
    )

    summarizer = SummarizerAgent(gemini_api_key=GEMINI_API_KEY)
    evaluator = EvaluatorAgent(gemini_api_key=GEMINI_API_KEY)

    designer = DesignerAgent()
    report_writer = ReportWriterAgent()
    rag=ResearchRAG(gemini_api_key=GEMINI_API_KEY)

    topic = req.topic.strip()
    if not topic:
        return {"status": "error", "message": "Topic cannot be empty."}

    # Create a simple session_id like GUI does
    session_id = f"session-{int(time.time())}"
    PROCESS_PIPELINE[session_id] = 1 
    # ---------- 2. Retrieval ----------
    papers = retrieval.search_recent_papers(
        query=topic,
        years_back=req.years_back,
        min_results=req.max_papers,
    )

    if not papers:
        return {
            "status": "error",
            "message": "No papers retrieved from Semantic Scholar.",
        }
    
    rag.ingest_papers(papers)
    core_paper_ids= [p["paper_id"] for p in papers if p.get("paper_id")]
    stats = rag.get_coverage_stats()
    print(f"[RAG] Coverage: {stats['full_text_papers']}/{stats['total_papers_indexed']} full-text, "
      f"{stats['abstract_only_papers']} abstract-only, {stats['total_chunks']} chunks total")


    # Store raw papers in JSON memory file
    json_memory.store_papers(session_id, topic, papers)
    PROCESS_PIPELINE[session_id] = 2
    # Optional: dataset search (can be displayed but not required for pipeline)
    datasets = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        datasets = retrieval.search_datasets(topic, num_results=5)

    # ---------- 3. Summarization ----------
    analysis = summarizer.summarize(papers, rag=rag)
    PROCESS_PIPELINE[session_id]=3
    # ---------- 4. Evaluation ----------
    is_valid, evaluation_report, recommendations = evaluator.evaluate_analysis(
        topic, analysis, papers
    )
    PROCESS_PIPELINE[session_id]=4
    # ---------- 5. Persist analysis & evaluation to JSON ----------
    json_memory.store_analysis(session_id, analysis)
    json_memory.store_evaluation(session_id, evaluation_report)

    # ---------- 6. Sync into your tools.memory_store ----------
    _sync_analysis_to_memory_store(topic, papers, analysis)

    # ---------- 7. Run experiment designer ----------
    design_result = await designer.run(
        {
            "topic": topic,
        }
    )
    PROCESS_PIPELINE[session_id]=5
    # ---------- 8. Run report writer (Markdown + IEEE LaTeX) ----------
    report_markdown = await report_writer.run(
        {"topic": topic, "format": "markdown"}
    )
    save_report(topic,report_markdown['content'])
    report_latex = await report_writer.run(
        {"topic": topic, "format": "latex"}
    )
    save_report(topic,report_latex['content'])
    PROCESS_PIPELINE[session_id]=6
    
    return {
        "status": "ok",
        "topic": topic,
        "session_id": session_id,
        "retrieval": {
            "paper_count": len(papers),
            "dataset_count": len(datasets),
        },
        "analysis": analysis,
        "evaluation": evaluation_report,
        "is_valid": is_valid,
        "recommendations": recommendations,
        "experiment_design": design_result,
        "report_markdown": report_markdown,
        "report_latex": report_latex,
        "memory_files": {
            "memory_file": os.path.abspath(memory_file),
            "analysis_file": os.path.abspath(analysis_file),
        },
    }
    
#endpoint for downloading markdown
@router.post("/download")
async def download_report(req: ReportRequest):
    result = await report_writer.run({
        "topic": req.topic,
        "format": "markdown"
    })

    content = result["content"]
    filename = result["filename"]
    
    dirs= "outputs"
    os.makedirs(dirs,exist_ok=True)
    filepath = os.path.join(dirs,filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return FileResponse(
        filepath,
        media_type="text/markdown",
        filename=filename
    )
#endpoint for downloading zip file(overleaf)
@router.post("/download-zip")
async def download_overleaf(req: ReportRequest):

    import zipfile
    import requests

    result = await report_writer.run({
        "topic": req.topic,
        "format": "latex"
    })

    if result["status"] != "ok":
        return result

    tex_content = result["content"]

    dirs="outputs"
    os.makedirs(dirs,exist_ok=True)
    
    temp_dir = tempfile.mkdtemp()

    main_tex_path = os.path.join(temp_dir, "main.tex")
    with open(main_tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # Download IEEEtran.cls
    cls_url = "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls"
    cls_path = os.path.join(temp_dir, "IEEEtran.cls")

    r = requests.get(cls_url, timeout=10)
    with open(cls_path, "wb") as f:
        f.write(r.content)

    # Bib file
    bib_path = os.path.join(temp_dir, "refs.bib")
    with open(bib_path, "w") as f:
        f.write("% Auto bibliography\n")

    zip_name = f"{req.topic.replace(' ', '_')}_overleaf.zip"
    zip_path = os.path.join(dirs, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(main_tex_path, "main.tex")
        zipf.write(cls_path, "IEEEtran.cls")
        zipf.write(bib_path, "refs.bib")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_name
    )
#endpoint to save history 
@router.get("/history")
async def get_history():
    from tools.report_history import load_reports
    return {"history":load_reports()}
#progress tracker
@router.get("/progress/{id}")
def get_progress(id: str):
    return {
        "id":id,
        "step":PROCESS_PIPELINE.get(id,0)
    }