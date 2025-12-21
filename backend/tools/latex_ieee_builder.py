from typing import List, Dict, Optional

def _escape(text: str) -> str:
    """Escape LaTeX special chars for dynamic content (not for raw LaTeX blocks)."""
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "\\": r"\\",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def build_ieee_latex_report(
    topic: str,
    sections: Dict,
    citations: List[Dict],
    author_block: Optional[str] = None,
) -> str:
    """
    Build a full IEEE-conference style LaTeX document (IEEEtran).
    Optimized for Overleaf/PDF use.
    """

    # default dummy author block if user doesn't supply
    if not author_block:
        author_block = (
            r"Author One, Author Two% <-this % stops a space \\" "\n"
            r"Department of Aerospace Engineering \\" "\n"
            r"Institute of Advanced Research \\" "\n"
            r"City, Country \\" "\n"
            r"{\tt\small author1@example.com, author2@example.com} \\ \\" "\n"
            r"Advisor: Dr. Advisor Name \\" "\n"
            r"Department of Aerospace Engineering \\" "\n"
            r"Institute of Advanced Research \\" "\n"
            r"City, Country \\" "\n"
            r"{\tt\small advisor@example.com}"
        )

    tex: List[str] = []

    # ------------------------------------------------------------------
    # Preamble and class  (SWITCHED TO IEEEtran)
    # ------------------------------------------------------------------
    tex.append(r"\documentclass[conference]{IEEEtran}")
    tex.append(r"\IEEEoverridecommandlockouts")  # ok for IEEEtran
    tex.append("")
    tex.append(r"% Packages")
    tex.append(r"\usepackage{graphics}")
    tex.append(r"\usepackage{epsfig}")
    tex.append(r"\usepackage{amsmath}")
    tex.append(r"\usepackage{amssymb}")
    tex.append(r"\usepackage{url}")
    tex.append(r"\usepackage[ruled, vlined, linesnumbered]{algorithm2e}")
    tex.append(r"\usepackage{verbatim}")
    tex.append(r"\usepackage{soul, color}")
    tex.append(r"\usepackage{lmodern}")
    tex.append(r"\usepackage{fancyhdr}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{fourier}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\usepackage{makecell}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{hyperref}")
    tex.append("")
    tex.append(r"\SetNlSty{large}{}{:}")
    tex.append(r"\renewcommand\theadalign{bc}")
    tex.append(r"\renewcommand\theadfont{\bfseries}")
    tex.append(r"\renewcommand\theadgape{\Gape[4pt]}")
    tex.append(r"\renewcommand\cellgape{\Gape[4pt]}")
    tex.append("")

    # ------------------------------------------------------------------
    # Title & author  (IEEEtran style)
    # ------------------------------------------------------------------
    tex.append(r"\title{" + _escape(topic) + r"}")
    tex.append("")
    tex.append(r"\author{" + author_block + r"}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append("")
    tex.append(r"\maketitle")
    tex.append("")

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    tex.append(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    tex.append(r"\begin{abstract}")
    if sections.get("abstract"):
        tex.append(_escape(sections["abstract"]) + r"\par")
    tex.append(r"\end{abstract}")
    tex.append("")

    # ------------------------------------------------------------------
    # Keywords (IEEEtran: IEEEkeywords)
    # ------------------------------------------------------------------
    tex.append(r"\begin{IEEEkeywords}")
    keywords = sections.get("keywords", [])
    if keywords:
        tex.append(_escape(", ".join(keywords)))
    else:
        tex.append(_escape(topic))

    tex.append(r"\end{IEEEkeywords}")
    tex.append("")
    tex.append(r"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # ------------------------------------------------------------------
    # INTRODUCTION
    # ------------------------------------------------------------------
    tex.append(r"\section{INTRODUCTION}")
    if sections.get("introduction"):
        tex.append(_escape(sections["introduction"]) + r"\par")
    tex.append("")

    # ------------------------------------------------------------------
    # LITERATURE SURVEY
    # ------------------------------------------------------------------
    tex.append(r"\section{LITERATURE SURVEY}")
    literature = sections.get("literature_review")
    if literature:
            if isinstance(literature, list):
                for entry in literature:
                    title = _escape(entry.get("title", "Related Study"))
                    tex.append(r"\subsection{" + title + r"}")
                    tex.append(_escape(entry.get("summary", "")) + r"\par")
            else:
                tex.append(_escape(literature) + r"\par")

    # ------------------------------------------------------------------
    # RESEARCH GAPS
    # ------------------------------------------------------------------
    tex.append(r"\section{RESEARCH GAPS}")
    tex.append(
        _escape(
            "Based on the literature survey, the following open research gaps are identified:"
        )
        + r"\par"
    )
    gaps = sections.get("research_gaps") or sections.get("gaps")
    tex.append(r"\begin{itemize}")

    if isinstance(gaps, list):
        for g in gaps:
            tex.append(r"\item " + _escape(g))
    elif isinstance(gaps, str):
        tex.append(r"\item " + _escape(gaps))
    else:
        tex.append(r"\item No explicit gaps were identified in the surveyed literature.")

    tex.append(r"\end{itemize}")
    tex.append("")

    # ------------------------------------------------------------------
    # PROPOSED EXPERIMENT
    # ------------------------------------------------------------------
    tex.append(r"\section{PROPOSED EXPERIMENT}")
    
    datasets = sections.get("datasets", [])
    tex.append(r"\subsection{Datasets}")
    tex.append(r"\begin{itemize}")

    if datasets:
        for d in datasets:
            tex.append(r"\item " + _escape(d))
    else:
        tex.append(r"\item No datasets were explicitly reported in the surveyed literature.")

    tex.append(r"\end{itemize}")
    tex.append("")
    
    baselines = sections.get("baselines", [])
    tex.append(r"\subsection{Baseline Methods}")
    tex.append(r"\begin{itemize}")

    if baselines:
        for b in baselines:
            tex.append(r"\item " + _escape(b))
    else:
        tex.append(r"\item No baseline methods were consistently identified.")

    tex.append(r"\end{itemize}")

    tex.append("")

    # ------------------------------------------------------------------
    # METHODOLOGY
    # ------------------------------------------------------------------
    tex.append(r"\section{PROPOSED METHODOLOGY}")

    methodology = sections.get("methodology", "")
    if methodology:
        tex.append(_escape(methodology) + r"\par")
    else:
        tex.append(
            _escape(
                "Based on the identified research gaps, a structured experimental and numerical methodology is proposed. "
                "However, detailed methodological formulation remains a subject of future work."
            ) + r"\par"
        )
    # ------------------------------------------------------------------
    # CONCLUSION
    # ------------------------------------------------------------------
    tex.append(r"\section{CONCLUSIONS}")
    tex.append(_escape(sections.get("conclusion","")) + r"\par")
    tex.append("")

    # ------------------------------------------------------------------
    # APPENDIX
    # ------------------------------------------------------------------
    tex.append(r"\section*{APPENDIX}")
    tex.append(
        _escape(
            "The appendix can include extended experimental settings, additional plots, or numerical tables. "
            "In the current draft, this section acts as a placeholder."
        )
        + r"\par"
    )
    tex.append("")

    # ------------------------------------------------------------------
    # REFERENCES
    # ------------------------------------------------------------------
    tex.append(r"\begin{thebibliography}{99}")
    if citations:
        for i, c in enumerate(citations, start=1):
            authors = _escape(", ".join(c.get("authors", []))) or "Unknown Authors"
            title = _escape(c.get("title", "Untitled"))
            year = _escape(str(c.get("year", "n.d.")))
            venue = _escape(c.get("journal", c.get("venue", "")))
            doi = _escape(c.get("doi", "")) if c.get("doi") else ""
            line = rf"\bibitem{{c{i}}} {authors}, ``{title},'' {venue}, {year}."
            if doi:
                line += f" DOI: {doi}."
            tex.append(line)
    else:
        tex.append(
            r"\bibitem{dummy} References will be populated once structured citation metadata is available."
        )
    tex.append(r"\end{thebibliography}")

    tex.append(r"\end{document}")

    return "\n".join(tex)
