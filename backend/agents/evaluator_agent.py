import google.generativeai as genai
import json
import re
from datetime import datetime


class EvaluatorAgent:
    def __init__(self, gemini_api_key, model_name="gemini-2.5-flash-lite"):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Thresholds for quality control
        self.min_relevance_score = 3.0
        self.min_completeness_score = 3.0
        self.min_novelty_score = 2.5
        self.min_overall_score = 3.0
        
    def evaluate_analysis(self, query, analysis_result, papers):
        """
        Evaluate the quality of summarization results
        Returns: (is_valid, evaluation_report, recommendations)
        """
        print(f"\n{'='*80}")
        print(" EVALUATOR AGENT: Starting evaluation...")
        print(f"{'='*80}\n")
        
        # Evaluate overall analysis
        overall_eval = self._evaluate_overall_quality(query, analysis_result)
        
        # Evaluate individual paper summaries
        paper_evals = []
        for idx, (paper, summary) in enumerate(zip(papers, analysis_result.get('per_paper', [])), 1):
            eval_result = self._evaluate_single_summary(query, paper, summary, idx)
            paper_evals.append(eval_result)
        
        # Calculate aggregate scores
        avg_relevance = sum(e['scores']['relevance'] for e in paper_evals) / len(paper_evals) if paper_evals else 0
        avg_completeness = sum(e['scores']['completeness'] for e in paper_evals) / len(paper_evals) if paper_evals else 0
        avg_novelty = sum(e['scores']['novelty'] for e in paper_evals) / len(paper_evals) if paper_evals else 0
        
        # Determine if analysis passes quality threshold
        is_valid = (
            overall_eval['scores']['overall'] >= self.min_overall_score and
            avg_relevance >= self.min_relevance_score and
            avg_completeness >= self.min_completeness_score and
            avg_novelty >= self.min_novelty_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_valid, overall_eval, paper_evals, avg_relevance, avg_completeness, avg_novelty
        )
        
        # Create evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'is_valid': is_valid,
            'overall_evaluation': overall_eval,
            'paper_evaluations': paper_evals,
            'aggregate_scores': {
                'avg_relevance': round(avg_relevance, 2),
                'avg_completeness': round(avg_completeness, 2),
                'avg_novelty': round(avg_novelty, 2)
            },
            'recommendations': recommendations
        }
        
        self._print_evaluation_summary(evaluation_report)
        
        return is_valid, evaluation_report, recommendations
    
    def _evaluate_single_summary(self, query, paper, summary, paper_num):
        """Evaluate a single paper summary"""
        print(f" Evaluating Paper {paper_num}: {paper.get('title', 'Unknown')[:60]}...")
        
        prompt = f"""
You are an expert research evaluator. Evaluate the following paper summary on these criteria:

Original Query: {query}

Paper Title: {paper.get('title', '')}
Paper Abstract: {(paper.get('abstract') or '')[:500]}

Generated Summary: {summary.get('summary', '')}
Methods Extracted: {', '.join(summary.get('methods', []))}
Gaps Identified: {', '.join(summary.get('gaps', []))}
Datasets Mentioned: {', '.join(summary.get('datasets', []))}

Evaluate on a scale of 1-5 for each criterion:
1. RELEVANCE: How relevant is this paper to the query "{query}"?
2. COMPLETENESS: Does the summary capture all key information (methods, gaps, datasets)?
3. ACCURACY: Does the summary accurately reflect the paper's content?
4. NOVELTY: Does this paper present novel approaches or insights?

Respond ONLY in this JSON format:
{{
    "relevance": <score 1-5>,
    "completeness": <score 1-5>,
    "accuracy": <score 1-5>,
    "novelty": <score 1-5>,
    "reasoning": {{
        "relevance": "<brief explanation>",
        "completeness": "<brief explanation>",
        "accuracy": "<brief explanation>",
        "novelty": "<brief explanation>"
    }},
    "issues": ["<issue1>", "<issue2>"],
    "strengths": ["<strength1>", "<strength2>"]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Parse JSON response
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                eval_data = json.loads(match.group(0))
                
                return {
                    'paper_num': paper_num,
                    'paper_title': paper.get('title', 'Unknown'),
                    'scores': {
                        'relevance': eval_data.get('relevance', 3),
                        'completeness': eval_data.get('completeness', 3),
                        'accuracy': eval_data.get('accuracy', 3),
                        'novelty': eval_data.get('novelty', 3)
                    },
                    'reasoning': eval_data.get('reasoning', {}),
                    'issues': eval_data.get('issues', []),
                    'strengths': eval_data.get('strengths', [])
                }
        except Exception as e:
            print(f" Error evaluating paper {paper_num}: {e}")
        
        # Return default scores if evaluation fails
        return {
            'paper_num': paper_num,
            'paper_title': paper.get('title', 'Unknown'),
            'scores': {'relevance': 3, 'completeness': 3, 'accuracy': 3, 'novelty': 3},
            'reasoning': {},
            'issues': ['Evaluation failed'],
            'strengths': []
        }
    
    def _evaluate_overall_quality(self, query, analysis_result):
        """Evaluate the overall analysis quality"""
        print(" Evaluating overall analysis quality...")
        
        prompt = f"""
You are an expert research evaluator. Evaluate the overall quality of this research analysis:

Original Query: {query}

Research Gaps Identified: {len(analysis_result.get('collected_gaps', []))} gaps
Methods Collected: {len(analysis_result.get('collected_methods', []))} methods
Datasets Collected: {len(analysis_result.get('collected_datasets', []))} datasets
Baselines Collected: {len(analysis_result.get('collected_baselines', []))} baselines

Sample Gaps: {'; '.join(analysis_result.get('collected_gaps', [])[:3])}
Sample Methods: {'; '.join(analysis_result.get('collected_methods', [])[:3])}

Proposed Methodology: {analysis_result.get('proposed_methodology', '')[:300]}

Evaluate on a scale of 1-5:
1. COMPREHENSIVENESS: Are enough papers analyzed with sufficient detail?
2. COHERENCE: Do the collected findings make sense together?
3. USEFULNESS: Is the proposed methodology practical and novel?
4. OVERALL: Overall quality of the analysis

Respond ONLY in this JSON format:
{{
    "comprehensiveness": <score 1-5>,
    "coherence": <score 1-5>,
    "usefulness": <score 1-5>,
    "overall": <score 1-5>,
    "reasoning": {{
        "comprehensiveness": "<explanation>",
        "coherence": "<explanation>",
        "usefulness": "<explanation>",
        "overall": "<explanation>"
    }},
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                eval_data = json.loads(match.group(0))
                return {
                    'scores': {
                        'comprehensiveness': eval_data.get('comprehensiveness', 3),
                        'coherence': eval_data.get('coherence', 3),
                        'usefulness': eval_data.get('usefulness', 3),
                        'overall': eval_data.get('overall', 3)
                    },
                    'reasoning': eval_data.get('reasoning', {}),
                    'strengths': eval_data.get('strengths', []),
                    'weaknesses': eval_data.get('weaknesses', [])
                }
        except Exception as e:
            print(f" Error in overall evaluation: {e}")
        
        return {
            'scores': {'comprehensiveness': 3, 'coherence': 3, 'usefulness': 3, 'overall': 3},
            'reasoning': {},
            'strengths': [],
            'weaknesses': ['Evaluation failed']
        }
    
    def _generate_recommendations(self, is_valid, overall_eval, paper_evals, avg_rel, avg_comp, avg_nov):
        """Generate actionable recommendations based on evaluation"""
        recommendations = {
            'action': 'ACCEPT' if is_valid else 'REFINE',
            'retrieval_refinements': [],
            'summarization_refinements': [],
            'general_improvements': []
        }
        
        if not is_valid:
            # Check what needs improvement
            if avg_rel < self.min_relevance_score:
                recommendations['retrieval_refinements'].append({
                    'issue': 'Low relevance scores',
                    'action': 'Refine search query to get more relevant papers',
                    'priority': 'HIGH'
                })
            
            if avg_comp < self.min_completeness_score:
                recommendations['summarization_refinements'].append({
                    'issue': 'Incomplete summaries',
                    'action': 'Enhance prompts to extract more detailed information',
                    'priority': 'HIGH'
                })
            
            if avg_nov < self.min_novelty_score:
                recommendations['retrieval_refinements'].append({
                    'issue': 'Low novelty scores',
                    'action': 'Filter for more recent papers or papers with novel methodologies',
                    'priority': 'MEDIUM'
                })
            
            if overall_eval['scores']['overall'] < self.min_overall_score:
                recommendations['general_improvements'].append({
                    'issue': 'Overall quality below threshold',
                    'action': 'Consider retrieving more papers or from different sources',
                    'priority': 'HIGH'
                })
        
        # Add specific paper-level recommendations
        low_quality_papers = [e for e in paper_evals if any(s < 3 for s in e['scores'].values())]
        if low_quality_papers:
            recommendations['general_improvements'].append({
                'issue': f'{len(low_quality_papers)} papers have low quality scores',
                'action': f'Review papers: {", ".join([str(p["paper_num"]) for p in low_quality_papers])}',
                'priority': 'MEDIUM'
            })
        
        return recommendations
    
    def _print_evaluation_summary(self, report):
        """Print a formatted evaluation summary"""
        print(f"\n{'='*80}")
        print(" EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        print(f" RESULT: {'PASSED' if report['is_valid'] else '❌ FAILED'}")
        print(f"Query: {report['query']}\n")
        
        print(" AGGREGATE SCORES:")
        agg = report['aggregate_scores']
        print(f"  • Average Relevance: {agg['avg_relevance']}/5.0")
        print(f"  • Average Completeness: {agg['avg_completeness']}/5.0")
        print(f"  • Average Novelty: {agg['avg_novelty']}/5.0")
        
        overall = report['overall_evaluation']['scores']
        print(f"\n  • Overall Quality: {overall['overall']}/5.0")
        print(f"  • Comprehensiveness: {overall['comprehensiveness']}/5.0")
        print(f"  • Coherence: {overall['coherence']}/5.0")
        print(f"  • Usefulness: {overall['usefulness']}/5.0\n")
        
        if not report['is_valid']:
            print(" RECOMMENDATIONS:")
            recs = report['recommendations']
            
            if recs['retrieval_refinements']:
                print("\n   Retrieval Refinements:")
                for rec in recs['retrieval_refinements']:
                    print(f"    [{rec['priority']}] {rec['issue']}: {rec['action']}")
            
            if recs['summarization_refinements']:
                print("\n   Summarization Refinements:")
                for rec in recs['summarization_refinements']:
                    print(f"    [{rec['priority']}] {rec['issue']}: {rec['action']}")
            
            if recs['general_improvements']:
                print("\n   General Improvements:")
                for rec in recs['general_improvements']:
                    print(f"    [{rec['priority']}] {rec['issue']}: {rec['action']}")
        
        print(f"\n{'='*80}\n")