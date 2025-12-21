from pydantic import Field
from google.adk.agents import Agent

from tools import memory_store
from tools.markdown_builder import build_markdown_report
from tools.latex_ieee_builder import build_ieee_latex_report


class ReportWriterAgent(Agent):
    store: str = Field(default="in-process-memory")

    def __init__(self, **kwargs):
        super().__init__(name="report_writer", **kwargs)

    async def run(self, input_json):
        topic = input_json["topic"]
        mode = input_json.get("format", "markdown")

        section_resp = memory_store.get_section_content(topic)
        if section_resp["status"] != "ok":
            return {
                "status": "error",
                "message": "Section-wise content not found. Run summarizer first."
            }

        sections = section_resp["section_content"]
        # Citations 
        citations = memory_store.get_citations(topic).get("citations", [])
        
        if mode == "markdown":
            content = build_markdown_report(
                topic=topic,
                sections=sections,
                citations=citations,
            )
            return {
                "status": "ok",
                "topic": topic,
                "format": "markdown",
                "content": content,
                "filename": f"{topic.replace(' ', '_')}.md",
            }

        elif mode == "latex":
            content = build_ieee_latex_report(
                topic=topic,
                sections=sections,
                citations=citations,
            )
            return {
                "status": "ok",
                "topic": topic,
                "format": "latex",
                "content": content,
                "filename": f"{topic.replace(' ', '_')}.tex",
            }

        else:
            return {
                "status": "error",
                "message": f"Unknown format '{mode}'"
            }
