import google.generativeai as genai
import json
import re

class SummarizerAgent:
    def __init__(self, gemini_api_key, model_name="gemini-2.5-flash-lite"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def summarize(self, paper_list):
        summaries = []
        all_gaps = []
        all_methods = []
        all_datasets = []
        all_baselines = []
        all_citations = []

        for paper in paper_list:
            result = self._summarize_single_paper(paper)
            summaries.append(result)
            all_gaps.extend(result.get("gaps", []))
            all_methods.extend(result.get("methods", []))
            all_datasets.extend(result.get("datasets", []))
            all_baselines.extend(result.get("baselines", []))
            all_citations.extend(result.get("citations", []))

        # Remove duplicates
        all_gaps = list({g.strip() for g in all_gaps if g.strip()})
        all_methods = list({m.strip() for m in all_methods if m.strip()})
        all_datasets = list({d.strip() for d in all_datasets if d.strip()})
        all_baselines = list({b.strip() for b in all_baselines if b.strip()})
        all_citations = list({c.strip() for c in all_citations if c.strip()})

        proposed_methodology = self._propose_methodology(all_gaps, all_methods, all_baselines, all_datasets)

        return {
            "per_paper": summaries,
            "collected_gaps": all_gaps,
            "collected_baselines": all_baselines,
            "collected_methods": all_methods,
            "collected_datasets": all_datasets,
            "collected_citations": all_citations,
            "proposed_methodology": proposed_methodology,
        }

    def _summarize_single_paper(self, paper):
        prompt = f"""
Analyze the following research paper's fields and provide structured JSON output as below.

Fields in focus: summary, methods, baselines, datasets, research gaps.

Title: {paper.get('title', '')}
Authors: {', '.join(paper.get('authors', []))}
Year: {paper.get('publication_year', '')}
Abstract: {paper.get('abstract', '')}
TLDR: {paper.get('tldr', '')}

JSON format only:
{{
    "summary": "<2-3 sentence summary>",
    "methods": ["..."],
    "baselines": ["..."],
    "datasets": ["..."],
    "gaps": ["..."],
    "citations": ["..."]
}}
ONLY answer in JSON.
"""
        try:
            response = self.model.generate_content(prompt)
            content = response.text

            # Try to find JSON in the response
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            # Fallback: try to parse from first { onwards
            start = content.find("{")
            if start != -1:
                part = content[start:]
                return json.loads(part)
            
            # If no JSON found, return error structure
            raise ValueError("No JSON found in response")
            
        except Exception as e:
            print(f"Error processing paper '{paper.get('title', 'Unknown')}': {str(e)}")
            return {
                "summary": f"Error processing: {paper.get('title', 'Unknown')}",
                "methods": [],
                "baselines": [],
                "datasets": [],
                "gaps": [],
                "error": str(e)
            }

    def _propose_methodology(self, all_gaps, all_methods, all_baselines, all_datasets):
        prompt = f"""
Based on the following extracted research gaps, existing methods, baselines, and datasets from a collection of recent research papers,
suggest a *unique new research methodology* (as if you were proposing your own research paper) that addresses as many of the open gaps as possible.

List of research gaps (bulleted):
{chr(10).join('- '+g for g in all_gaps) if all_gaps else "- No specific gaps identified"}

Existing methods:
{', '.join(all_methods) if all_methods else "None identified"}

Baselines:
{', '.join(all_baselines) if all_baselines else "None identified"}

Datasets:
{', '.join(all_datasets) if all_datasets else "None identified"}

Respond with a concise paragraph describing a unique, innovative research methodology that could be the next step for the field, grounded in the above.

ONLY respond with the paragraph, NO JSON or lists.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
                
        except Exception as e:
            print(f"Error generating methodology: {str(e)}")
            return f"Error generating proposed methodology: {str(e)}"