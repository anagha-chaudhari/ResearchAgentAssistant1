#core memory bank where the write and read op of summary and everything occurs

from typing import Dict, Any, List

# Simple in-memory DB, process-wide
_MEMORY_DB: Dict[str, Dict[str, Any]] = {}


def _get_topic_bucket(topic: str) -> Dict[str, Any]:
    """
    Ensure a bucket exists for a given topic and return it.
    """
    if topic not in _MEMORY_DB:
        _MEMORY_DB[topic] = {
            "summaries": [],
            "gaps": [],
            "citations": [],
            "experiment_plan": None,
        }
    return _MEMORY_DB[topic]


# ---------- WRITE OPERATIONS ----------

def save_summaries(topic: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket = _get_topic_bucket(topic)
    bucket["summaries"] = summaries

    # Optional: auto-collect gaps & citations from summaries
    gaps = []
    citations = []
    for s in summaries:
        gaps.extend(s.get("gaps", []))
        c = s.get("citations")
        if c:
            citations.append(c)

    if gaps:
        bucket["gaps"] = gaps
    if citations:
        bucket["citations"] = citations

    return {"status": "ok"}


def save_gaps(topic: str, gaps: List[str]) -> Dict[str, Any]:
    bucket = _get_topic_bucket(topic)
    bucket["gaps"] = gaps
    return {"status": "ok"}


def save_citations(topic: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket = _get_topic_bucket(topic)
    bucket["citations"] = citations
    return {"status": "ok"}


def save_experiment_plan(topic: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    bucket = _get_topic_bucket(topic)
    bucket["experiment_plan"] = plan
    return {"status": "ok"}


# ---------- READ OPERATIONS ----------

def get_summaries(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["summaries"]:
        return {"status": "not_found", "summaries": []}
    return {"status": "ok", "summaries": bucket["summaries"]}


def get_gaps(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["gaps"]:
        return {"status": "not_found", "gaps": []}
    return {"status": "ok", "gaps": bucket["gaps"]}


def get_citations(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["citations"]:
        return {"status": "not_found", "citations": []}
    return {"status": "ok", "citations": bucket["citations"]}


def get_experiment_plan(topic: str) -> Dict[str, Any]:
    bucket = _MEMORY_DB.get(topic)
    if not bucket or not bucket["experiment_plan"]:
        return {"status": "not_found", "experiment_plan": None}
    return {"status": "ok", "experiment_plan": bucket["experiment_plan"]}
