def build_markdown_report(topic, summaries, gaps, citations, experiment_plan):
    md = []

    # Title
    md.append(f"# {topic}\n")
    md.append("### Research Report\n")
    md.append("---\n")

    # Abstract
    md.append("## 1. Abstract\n")
    md.append(
        "This paper presents a structured synthesis of prior research on the topic, identifies critical research gaps, "
        "and proposes an experiment to advance understanding and improve Thermal Protection System (TPS) performance.\n"
    )

    # Introduction
    md.append("## 2. Introduction\n")
    md.append(
        "This section presents an overview of the research papers relevant to the topic.\n"
    )
    for idx, p in enumerate(summaries, start=1):
        md.append(f"### 2.{idx} {p.get('title','Untitled')}\n")
        md.append(f"{p.get('summary','No summary available.')}\n\n")

    # Literature Review
    md.append("## 3. Literature Review\n")
    for idx, p in enumerate(summaries, start=1):
        md.append(f"### 3.{idx} {p.get('title','Untitled')}\n")
        md.append(f"- **Methods Used**: {', '.join(p.get('methods', []))}\n")
        md.append(f"- **Key Findings**: {', '.join(p.get('key_findings', []))}\n")
        md.append(f"- **Limitations**: {', '.join(p.get('limitations', []))}\n\n")


    # Problem Statement
    md.append("## 4. Problem Statement\n")
    md.append(
        "Thermal Protection Systems (TPS) face challenges in balancing mass efficiency, thermal stability, "
        "and structural integrity during high-enthalpy re-entry.\n"
    )

    # Research Gaps
    md.append("## 5. Research Gaps\n")
    if not gaps:
        md.append("- no explicit gaps identified.\n")
    else:
        for g in gaps:
            md.append(f"- {g}\n")
    md.append("\n")

    # Proposed Experiment
    md.append("## 6. Proposed Experiment\n")
    md.append(f"### Hypothesis\n{experiment_plan['hypothesis']}\n\n")

    # Datasets
    md.append("### 6.1 Datasets\n")
    if not experiment_plan["datasets"]:
        md.append("-no datasets available\n")
    else:
        for d in experiment_plan["datasets"]:
            link = d.get("link","will define soon")
            md.append(f"- **{d['name']}** — {d['description']} *(Link: {link})*\n")

    # Evaluation Metrics
    md.append("\n### 6.2 Evaluation Metrics\n")
    for m in experiment_plan["evaluation_metrics"]:
        md.append(f"- **{m['name']}** — {m['interpretation']}\n")

    # Baselines
    md.append("\n### 6.3 Baseline Methods\n")
    for b in experiment_plan["baseline_methods"]:
        md.append(f"- **{b['name']}** — {b['reason']}\n")

    # Methodology
    md.append("\n## 7. Methodology\n")
    md.append(
        "The experiment will use a high-fidelity thermal Finite Element (FEA) simulation "
        "with validated material datasets for PICA and CMC.\n"
    )

    # Expected Results
    md.append("\n## 8. Expected Results\n")
    md.append(
        "The hybrid PICA–CMC TPS structure is expected to reduce peak thermal stress and "
        "outperform baseline TPS configurations.\n"
    )

    # Implementation Notes
    md.append("\n## 9. Implementation Notes\n")
    impl = experiment_plan.get("implementation_notes", {})

    seed = impl.get("seed", "N/A")
    env = impl.get("env", "N/A")

    md.append(f"- Seed: {seed}\n")
    md.append(f"- Environment: {env}\n")

    # Safety Notes
    s=experiment_plan.get("safety",{})
    risk=s.get("risk_domains",[])
    notes=s.get("notes","N/A")
    md.append("\n## 10. Safety Notes\n")
    md.append(f"- **Domain Risk**: {risk}\n")
    md.append(f"- **Notes**: {notes}\n")

    # References
    md.append("\n## 11. References\n")
    for c in citations:
        a = c.get("authors",["unknown"])[0]
        md.append(
            f"- {a} et al., *{c.get('title','Untitled')}*, {c.get('year','n.d.')}. DOI: {c.get('doi','N/A')}\n"
        )

    return "\n".join(md)
