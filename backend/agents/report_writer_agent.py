from pydantic import Field
from google.adk.agents import Agent

from tools import memory_store
from tools.markdown_builder import build_markdown_report
# from tools.latex_builder import build_latex_report
from tools.latex_ieee_builder import build_ieee_latex_report

class ReportWriterAgent(Agent):
    store: str = Field(default="in-process-memory")
    
    def __init__(self,**kwargs):
        super().__init__(name="report_writer",**kwargs)
        
    async def run(self, input_json):
        topic=input_json['topic']
        mode=input_json.get("format","markdown")
        
        summaries=memory_store.get_summaries(topic).get('summaries',[])
        gaps=memory_store.get_gaps(topic).get('gaps',[])
        citations=memory_store.get_citations(topic).get('citations',[])
        experiment_plan=memory_store.get_experiment_plan(topic).get('experiment_plan',None)
        
        if not experiment_plan:
            return{"status":"error","message":"not found.run designer_agent.py"}
        
        if mode == "markdown":
            md = build_markdown_report(
                topic=topic,
                summaries=summaries,
                gaps=gaps,
                citations=citations,
                experiment_plan=experiment_plan,
            )

            return {
                "status": "ok",
                "topic": topic,
                "format": "markdown",
                "content": md,
                "filename": f"{topic.replace(' ', '_')}.md",
            }

        elif mode == "latex":
            tex = build_ieee_latex_report(
                topic=topic,
                summaries=summaries,
                gaps=gaps,
                citations=citations,
                experiment_plan=experiment_plan,
            )

            return {
                "status": "ok",
                "topic": topic,
                "format": "latex",
                "content": tex,
                "filename": f"{topic.replace(' ', '_')}.tex",
            }

        else:
            return {"status": "error", "message": f"Unknown format '{mode}'"}