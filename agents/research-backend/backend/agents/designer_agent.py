from google.adk.agents import Agent
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from pydantic import Field

from tools import memory_store


class DesignerAgent(Agent):
    memory: InMemoryMemoryService = Field(default_factory=InMemoryMemoryService)
    session: InMemorySessionService = Field(default_factory=InMemorySessionService)

    def __init__(self, **kwargs):
        super().__init__(name="experiment_designer", **kwargs)

    async def run(self, input_json):
        topic = input_json["topic"]
        task = input_json.get("task_type", "thermal_simulation")
        
        mem_summaries = memory_store.get_summaries(topic)
        mem_gaps = memory_store.get_gaps(topic)
        mem_citations = memory_store.get_citations(topic)

        if mem_summaries["status"] != "ok":
            return {
                "status": "error",
                "message": "No summaries found in memory. Run retrieval + summarization first."
            }

        summaries = mem_summaries["summaries"]

        if mem_gaps["status"] == "ok":
            gaps = mem_gaps["gaps"]
        else:
            gaps = []
            for s in summaries:
                gaps.extend(s.get("gaps", []))

        datasets = [
            {
                "name": "Default TPS Dataset",
                "description": "Placeholder dataset derived from summarized literature.",
                "source": "Derived from summarized papers"
            }
        ]

        evaluation_metrics = [
            {"name": "Max Temperature", "interpretation": "Lower is better"},
            {"name": "Ablation Rate", "interpretation": "Lower is better"},
            {"name": "Thermal Stress", "interpretation": "Lower is better"},
        ]

        baseline_methods = [
            {"name": "PICA", "reason": "Standard TPS baseline"},
            {"name": "AVCOAT", "reason": "Heritage Apollo TPS"},
        ]

        plan = {
            "topic": topic,
            "task_type": task,
            "gaps": gaps,
            "hypothesis": (
                "A hybrid TPS material combining PICA and CMC layers will "
                "reduce peak thermal stress by at least 8% compared to PICA alone."
            ),
            "datasets": datasets,
            "evaluation_metrics": evaluation_metrics,
            "baseline_methods": baseline_methods,
            "implementation_notes": {
                "seed": 42,
                "environment": "Python + thermal simulation framework"
            },
            "citations_used": mem_citations.get("citations", [])
        }

        memory_store.save_experiment_plan(topic, plan)

        return {
            "status": "ok",
            "experiment_plan": plan
        }
