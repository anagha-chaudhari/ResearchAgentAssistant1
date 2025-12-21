from google.adk.agents import Agent
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from pydantic import Field
from typing import List, Dict, Any

from tools import memory_store


class DesignerAgent(Agent):
    memory: InMemoryMemoryService = Field(default_factory=InMemoryMemoryService)
    session: InMemorySessionService = Field(default_factory=InMemorySessionService)

    def __init__(self, **kwargs):
        super().__init__(name="experiment_designer", **kwargs)

    async def run(self, input_json):
        topic = input_json["topic"]
        
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

        # ========== EXTRACT FROM ACTUAL PAPERS ==========
        datasets = self._extract_datasets(summaries)
        evaluation_metrics = self._extract_metrics(summaries)
        baseline_methods = self._extract_baselines(summaries)
        hypothesis = self._generate_hypothesis(topic, gaps, summaries)

        plan = {
            "topic": topic,
            "gaps": gaps,
            "hypothesis": hypothesis,
            "datasets": datasets,
            "evaluation_metrics": evaluation_metrics,
            "baseline_methods": baseline_methods,
            "implementation_notes": {
                "seed": 42,
                "environment": "Python + relevant frameworks for this domain"
            },
            "citations_used": mem_citations.get("citations", [])
        }

        memory_store.save_experiment_plan(topic, plan)

        return {
            "status": "ok",
            "experiment_plan": plan
        }

    # ========== HELPER METHODS ==========

    def _generate_hypothesis(self, topic: str, gaps: List[str], summaries: List[Dict]) -> str:
        """Generate hypothesis from actual research gaps"""
        if gaps:
            first_gap = gaps[0][:200]  # Use first gap, truncate if too long
            return f"Addressing {first_gap} will improve outcomes in {topic} research and applications."
        
        # Fallback if no gaps
        return f"A comprehensive analysis of recent advances in {topic} will identify best practices and optimization opportunities."

    def _extract_datasets(self, summaries: List[Dict]) -> List[Dict]:
        """Extract datasets mentioned in papers"""
        datasets = []
        seen = set()
        
        for summary in summaries:
            methods = summary.get("methods", [])
            findings = summary.get("key_findings", [])
            
            # Look for dataset mentions
            for text in methods + findings:
                text_lower = text.lower()
                if any(kw in text_lower for kw in ["dataset", "corpus", "benchmark", "collection"]):
                    if text not in seen and len(text) < 150:
                        datasets.append({
                            "name": text.strip(),
                            "description": f"Dataset from: {summary.get('title', 'Unknown')[:50]}",
                            "source": "Extracted from literature"
                        })
                        seen.add(text)
        
        # Default if none found
        if not datasets:
            datasets.append({
                "name": "Domain-appropriate dataset",
                "description": "Dataset to be selected based on research requirements",
                "source": "To be identified from literature or public repositories"
            })
        
        return datasets[:3]  # Limit to 3

    def _extract_metrics(self, summaries: List[Dict]) -> List[Dict]:
        """Extract evaluation metrics from papers"""
        metrics = []
        seen = set()
        
        # Common metrics
        metric_map = {
            "accuracy": ("Accuracy", "Higher is better"),
            "precision": ("Precision", "Higher is better"),
            "recall": ("Recall", "Higher is better"),
            "f1": ("F1 Score", "Higher is better"),
            "auc": ("AUC", "Higher is better"),
            "rmse": ("RMSE", "Lower is better"),
            "mae": ("MAE", "Lower is better"),
            "loss": ("Loss", "Lower is better"),
            "error": ("Error Rate", "Lower is better"),
            "bleu": ("BLEU Score", "Higher is better"),
            "rouge": ("ROUGE Score", "Higher is better"),
        }
        
        for summary in summaries:
            findings = " ".join(summary.get("key_findings", []))
            methods = " ".join(summary.get("methods", []))
            combined = (findings + " " + methods).lower()
            
            for keyword, (name, interpretation) in metric_map.items():
                if keyword in combined and name not in seen:
                    metrics.append({
                        "name": name,
                        "interpretation": interpretation
                    })
                    seen.add(name)
        
        # Default metrics
        if not metrics:
            metrics = [
                {"name": "Primary Performance Metric", "interpretation": "Domain-specific evaluation"},
                {"name": "Secondary Quality Metric", "interpretation": "Domain-specific evaluation"}
            ]
        
        return metrics[:5]  # Limit to 5

    def _extract_baselines(self, summaries: List[Dict]) -> List[Dict]:
        """Extract baseline methods from papers"""
        baselines = []
        seen = set()
        
        for summary in summaries:
            methods = summary.get("methods", [])
            findings = summary.get("key_findings", [])
            
            for text in methods + findings:
                text_lower = text.lower()
                
                # Look for baseline indicators
                if any(kw in text_lower for kw in ["baseline", "compared", "benchmark", "existing", "prior method", "traditional"]):
                    if text not in seen and len(text) < 150:
                        baselines.append({
                            "name": text.strip(),
                            "reason": "Baseline from literature"
                        })
                        seen.add(text)
        
        # Default baseline
        if not baselines:
            baselines.append({
                "name": "Current state-of-the-art approach",
                "reason": "Standard baseline for comparison in this domain"
            })
        
        return baselines[:3]  # Limit to 3
