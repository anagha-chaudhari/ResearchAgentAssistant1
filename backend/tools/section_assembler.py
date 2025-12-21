
import re
from collections import Counter
from typing import List, Dict

STOPWORDS = {
    "the", "and", "of", "in", "to", "for", "with", "on", "by",
    "an", "is", "are", "this", "that", "as", "at", "from",
    "study", "paper", "analysis", "results", "using",
    "based", "approach", "method", "methods", "system"
}

# -----------------------------
def extract_keywords(texts: List[str], top_k: int = 10) -> List[str]:
    """
    Extract frequent domain-relevant keywords from text corpus.
    Uses frequency-based filtering (no embeddings, no LLMs).

    Args:
        texts: List of abstracts or section texts
        top_k: Number of keywords to return

    Returns:
        List of keywords sorted by relevance
    """
    tokens = []

    for text in texts:
        if not text:
            continue

        # Keep alphabetic tokens only, length >= 4
        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        tokens.extend(w for w in words if w not in STOPWORDS)

    freq = Counter(tokens)

    return [word for word, _ in freq.most_common(top_k)]

# -----------------------------
def build_abstract(topic: str, papers: List[Dict]) -> str:
    abstracts = [p["abstract"] for p in papers if p.get("abstract")]

    if not abstracts:
        return (
            f"This paper reviews recent research related to {topic.lower()}, "
            "focusing on experimental and numerical investigations reported in the literature."
        )

    keywords = extract_keywords(abstracts, top_k=5)

    core_focus = ", ".join(keywords[:3]) if keywords else "key phenomena"

    return (
        f"Recent studies on {topic.lower()} primarily investigate {core_focus}. "
        "This review synthesizes existing experimental and numerical findings, "
        "highlights methodological trends, and identifies unresolved challenges "
        "that motivate further research."
    )


# -----------------------------
def build_introduction(topic: str, papers: List[Dict], max_papers: int = 4) -> str:
    intro_paragraphs = []

    for p in papers[:max_papers]:
        title = p.get("title", "This study")
        abstract = p.get("abstract", "")

        snippet = abstract[:220].rstrip() + "..." if abstract else "addresses key aspects of the problem."

        intro_paragraphs.append(
            f"{title} examines {snippet}"
        )

    if not intro_paragraphs:
        return (
            f"The research area of {topic.lower()} has gained attention due to its "
            "importance in system safety, performance, and operational reliability."
        )

    return "\n\n".join(intro_paragraphs)

# -----------------------------
def build_literature_review(papers: List[Dict]) -> List[Dict]:
    review = []

    for p in papers:
        review.append({
            "title": p.get("title", "Untitled"),
            "year": p.get("publication_year", "n.d."),
            "summary": p.get("abstract", "No abstract available.")
        })

    return review


def assemble_sections(topic: str, papers: List[Dict]) -> Dict[str, object]:
    """
    Assemble section-wise content for review paper.
    This output should be stored in memory_store and consumed by ReportWriterAgent.
    """
    abstracts = [p["abstract"] for p in papers if p.get("abstract")]
    keywords = extract_keywords(abstracts, top_k=5)
     
    return {
        "abstract": build_abstract(topic, papers),
        "introduction": build_introduction(topic, papers),
        "literature_review": build_literature_review(papers),
        "keywords":keywords
    }
