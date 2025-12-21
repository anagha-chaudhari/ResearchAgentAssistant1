def build_markdown_report(topic: str, sections: dict, citations: list):
    md = []

    # ---------------- Title ----------------
    md.append(f"# {topic}\n")
    md.append("---\n")

    # ---------------- Abstract ----------------
    if sections.get("abstract"):
        md.append("## Abstract\n")
        md.append(sections.get("abstract").strip() + "\n\n")

    # ---------------- Introduction ----------------
    if sections.get("introduction"):
        md.append("## Introduction\n")
        md.append(sections.get("introduction").strip() + "\n\n")

    # ---------------- Literature Review ----------------
    if sections.get("literature_review"):
        md.append("## Literature Review\n")

        # Can be list (per-paper) OR string (synthesized)
        lr = ( sections.get("literature_review") or sections.get("literature"))

        if isinstance(lr, list):
            for idx, entry in enumerate(lr, start=1):
                title = entry.get("title", f"Study {idx}")
                summary = entry.get("summary", "")
                md.append(f"### {title}\n")
                md.append(summary.strip() + "\n\n")
        else:
            md.append(lr.strip() + "\n\n")

    # ---------------- Research Gaps ----------------
    if sections.get("research_gaps"):
        md.append("## Research Gaps\n")
        gaps = (sections.get("research_gaps") or sections.get("research"))

        if isinstance(gaps, list):
            for g in gaps:
                md.append(f"- {g}\n")
        else:
            md.append(gaps.strip() + "\n")
        md.append("\n")

    # ---------------- Methodology ----------------
    if sections.get("methodology"):
        md.append("## Methodology\n")
        md.append(sections.get("methodology").strip() + "\n\n")

    # ---------------- Results ----------------
    if sections.get("results"):
        md.append("## Results and Discussion\n")
        md.append(sections.get("results").strip() + "\n\n")

    # ---------------- Conclusion ----------------
    if sections.get("conclusion"):
        md.append("## Conclusion\n")
        md.append(sections.get("conclusion").strip() + "\n\n")

    # ---------------- References ----------------
    if citations:
        md.append("## References\n")
        for c in citations:
            authors = ", ".join(c.get("authors", ["Unknown"]))
            title = c.get("title", "Untitled")
            year = c.get("year", "n.d.")
            venue = c.get("journal", c.get("venue", ""))
            doi = c.get("doi", "")
            ref = f"- {authors}, *{title}*, {venue}, {year}"
            if doi:
                ref += f". DOI: {doi}"
            md.append(ref + "\n")

    return "\n".join(md)
