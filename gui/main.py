import tkinter as tk
from datetime import datetime
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.summarizer_agent import SummarizerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from dotenv import load_dotenv
import os

load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate that all required API keys are present
if not all([SEMANTIC_SCHOLAR_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, GEMINI_API_KEY]):
    raise ValueError("Missing required API keys in .env file!")

# === USER INTERFACE ===
class ResearchAssistantUI:
    def __init__(self, root, retrieval_agent, memory_agent):
        self.retrieval_agent = retrieval_agent
        self.memory_agent = memory_agent
        self.root = root
        self.current_session_id = None
        
        root.title("AI Research Assistant")
        root.geometry("1000x700")
        
        # Query input
        tk.Label(root, text="Enter Research Topic:", font=("Arial", 12, "bold")).pack(pady=10)
        self.query_var = tk.StringVar()
        self.entry = tk.Entry(root, width=60, textvariable=self.query_var, font=("Arial", 11))
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", self._on_search_papers)
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Search Papers", command=self._on_search_papers, 
                 width=15, bg="#4CAF50", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Search Datasets", command=self._on_search_datasets, 
                 width=15, bg="#2196F3", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Search Both", command=self._on_search_both, 
                 width=15, bg="#FF9800", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Memory Stats", command=self._show_memory_stats, 
                 width=15, bg="#9C27B0", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Summarize Papers", command=self._on_summarize, 
                 width=15, bg="#D32F2F", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to search")
        self.status_label = tk.Label(root, textvariable=self.status_var, 
                                     font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)
        
        # Results text box
        self.textbox = tk.Text(root, wrap=tk.WORD, height=30, width=110, font=("Consolas", 9))
        self.textbox.pack(pady=5, padx=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.textbox)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.textbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.textbox.yview)

    def _on_search_papers(self, event=None):
        topic = self.query_var.get().strip()
        if not topic:
            self.status_var.set("Please enter a research topic")
            return
        
        self.textbox.delete(1.0, tk.END)
        self.status_var.set(f"Searching for papers on '{topic}'...")
        self.root.update()
        
        # Retrieve papers
        papers = self.retrieval_agent.search_recent_papers(topic)
        
        if not papers:
            self.status_var.set("No papers found")
            return
        
        # Store in memory
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory_agent.store_papers(self.current_session_id, topic, papers)
        
        # Display
        self.status_var.set(f"Found {len(papers)} papers | Session: {self.current_session_id}")
        self._display_papers(papers)

    def _on_search_datasets(self):
        topic = self.query_var.get().strip()
        if not topic:
            self.status_var.set("Please enter a research topic")
            return
        
        self.textbox.delete(1.0, tk.END)
        self.status_var.set(f"Searching for datasets on '{topic}'...")
        self.root.update()
        
        datasets = self.retrieval_agent.search_datasets(topic)
        
        if not datasets:
            self.status_var.set("No datasets found")
            return
        
        self.status_var.set(f"Found {len(datasets)} datasets")
        self._display_datasets(datasets)

    def _on_search_both(self):
        topic = self.query_var.get().strip()
        if not topic:
            self.status_var.set("Please enter a research topic")
            return
        
        self.textbox.delete(1.0, tk.END)
        self.status_var.set(f"Searching for papers and datasets...")
        self.root.update()
        
        # Search papers
        papers = self.retrieval_agent.search_recent_papers(topic)
        
        # Store in memory
        if papers:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.memory_agent.store_papers(self.current_session_id, topic, papers)
        
        # Search datasets
        datasets = self.retrieval_agent.search_datasets(topic)
        
        self.status_var.set(f"{len(papers)} papers & {len(datasets)} datasets | Session: {self.current_session_id}")
        
        # Display
        if papers:
            self.textbox.insert(tk.END, "=" * 100 + "\n")
            self.textbox.insert(tk.END, "RESEARCH PAPERS\n")
            self.textbox.insert(tk.END, "=" * 100 + "\n\n")
            self._display_papers(papers)
        
        if datasets:
            self.textbox.insert(tk.END, "\n" + "=" * 100 + "\n")
            self.textbox.insert(tk.END, "RELEVANT DATASETS\n")
            self.textbox.insert(tk.END, "=" * 100 + "\n\n")
            self._display_datasets(datasets)
    
    def _on_summarize(self):
        """Summarize papers with automatic evaluation and refinement"""
        if not self.current_session_id:
            self.status_var.set("No session found. Retrieve papers first.")
            return
        
        papers = self.memory_agent.get_papers_by_session(self.current_session_id)
        if not papers:
            self.status_var.set("No papers found in the current session.")
            return
        
        # Get the query from session
        query = self.memory_agent.get_session_query(self.current_session_id)
        if not query:
            query = "research topic"
        
        self.status_var.set(f"Summarizing {len(papers)} papers...")
        self.root.update()
        
        max_retries = 2  # Allow one refinement attempt
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Step 1: Summarize
                print(f"\n Attempt {retry_count + 1}/{max_retries + 1}")
                summarizer = SummarizerAgent(gemini_api_key=GEMINI_API_KEY)
                summary_result = summarizer.summarize(papers)
                
                # Step 2: Evaluate
                self.status_var.set("🔍 Evaluating analysis quality...")
                self.root.update()
                
                evaluator = EvaluatorAgent(gemini_api_key=GEMINI_API_KEY)
                is_valid, evaluation_report, recommendations = evaluator.evaluate_analysis(
                    query, summary_result, papers
                )
                
                # Step 3: Store results
                self.memory_agent.store_analysis(self.current_session_id, summary_result)
                self.memory_agent.store_evaluation(self.current_session_id, evaluation_report)
                
                # Step 4: Check if we need to refine
                if is_valid:
                    print(" Analysis passed evaluation!")
                    self._display_successful_analysis(summary_result, evaluation_report)
                    overall_score = evaluation_report['overall_evaluation']['scores']['overall']
                    self.status_var.set(f" Analysis complete and validated! (Score: {overall_score}/5.0)")
                    break
                else:
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        print(f"⚠️ Analysis failed evaluation. Attempting refinement...")
                        self.status_var.set(f"⚠️ Quality insufficient. Refining... (Attempt {retry_count + 1})")
                        self.root.update()
                        
                        # Apply refinements based on recommendations
                        papers = self._apply_refinements(query, papers, recommendations)
                        
                        # Continue to next iteration
                        continue
                    else:
                        print(" Max retries reached. Displaying results with warnings.")
                        self._display_failed_analysis(summary_result, evaluation_report)
                        self.status_var.set("⚠️ Analysis complete but below quality threshold")
                        break
                
            except Exception as e:
                self.status_var.set(f"= Error: {str(e)}")
                print(f"Error in summarization pipeline: {e}")
                import traceback
                traceback.print_exc()
                break

    def _apply_refinements(self, query, papers, recommendations):
        """Apply refinements based on evaluator recommendations"""
        print("\n🔧 Applying refinements...")
        
        # Check if we need to retrieve more/better papers
        needs_retrieval_refinement = any(
            rec for rec in recommendations.get('retrieval_refinements', [])
            if rec['priority'] == 'HIGH'
        )
        
        if needs_retrieval_refinement:
            print(" Retrieving additional papers...")
            # Combine with existing papers (deduplicate)
            existing_ids = {p['paper_id'] for p in papers}
            
            # Refine search - get more recent or more specific papers
            additional_papers = self.retrieval_agent.search_recent_papers(
                query,
                years_back=4,
                min_results=5,
                exclude_ids=existing_ids
            )
            
            if additional_papers:
                print(f" Added {len(additional_papers)} new non-duplicate papers")
                papers.extend(additional_papers[:3])  # Add up to 3 new papers
                
                # Update memory with new papers
                self.memory_agent.store_papers(self.current_session_id, query, papers)
        
        return papers

    def _display_successful_analysis(self, summary_result, evaluation_report):
        """Display analysis that passed evaluation"""
        self.textbox.delete(1.0, tk.END)
        
        # Show validation badge
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        self.textbox.insert(tk.END, " VALIDATED RESEARCH ANALYSIS\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n\n")
        
        # Show evaluation scores
        agg = evaluation_report['aggregate_scores']
        self.textbox.insert(tk.END, " Quality Scores:\n")
        self.textbox.insert(tk.END, f"   Relevance: {agg['avg_relevance']}/5.0 | ")
        self.textbox.insert(tk.END, f"Completeness: {agg['avg_completeness']}/5.0 | ")
        self.textbox.insert(tk.END, f"Novelty: {agg['avg_novelty']}/5.0\n")
        
        overall = evaluation_report['overall_evaluation']['scores']
        self.textbox.insert(tk.END, f"   Overall: {overall['overall']}/5.0 | ")
        self.textbox.insert(tk.END, f"Comprehensiveness: {overall['comprehensiveness']}/5.0\n\n")
        
        # Display the actual analysis
        self._display_analysis_content(summary_result)

    def _display_failed_analysis(self, summary_result, evaluation_report):
        """Display analysis that failed evaluation with warnings"""
        self.textbox.delete(1.0, tk.END)
        
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        self.textbox.insert(tk.END, " RESEARCH ANALYSIS (Quality Warnings)\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n\n")
        
        # Show what failed
        self.textbox.insert(tk.END, " Quality Issues Detected:\n")
        recs = evaluation_report['recommendations']
        
        for rec in recs.get('retrieval_refinements', []):
            self.textbox.insert(tk.END, f"   [{rec['priority']}] {rec['issue']}\n")
        for rec in recs.get('summarization_refinements', []):
            self.textbox.insert(tk.END, f"   [{rec['priority']}] {rec['issue']}\n")
        for rec in recs.get('general_improvements', []):
            self.textbox.insert(tk.END, f"   [{rec['priority']}] {rec['issue']}\n")
        
        self.textbox.insert(tk.END, "\n")
        
        # Display the analysis anyway
        self._display_analysis_content(summary_result)

    def _display_analysis_content(self, summary_result):
        """Display the actual analysis content"""
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        self.textbox.insert(tk.END, " PER-PAPER SUMMARIES\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n\n")
        
        for idx, summary in enumerate(summary_result.get("per_paper", []), 1):
            self.textbox.insert(tk.END, f"[Paper {idx}]\n")
            self.textbox.insert(tk.END, f"Summary: {summary.get('summary', 'N/A')}\n")
            
            if summary.get('methods'):
                self.textbox.insert(tk.END, f"Methods: {', '.join(summary['methods'])}\n")
            if summary.get('baselines'):
                self.textbox.insert(tk.END, f"Baselines: {', '.join(summary['baselines'])}\n")
            if summary.get('datasets'):
                self.textbox.insert(tk.END, f"Datasets: {', '.join(summary['datasets'])}\n")
            if summary.get('gaps'):
                self.textbox.insert(tk.END, f"Gaps: {', '.join(summary['gaps'])}\n")
            if summary.get('error'):
                self.textbox.insert(tk.END, f" Error: {summary['error']}\n")
            
            self.textbox.insert(tk.END, "-" * 90 + "\n\n")
        
        # Collected findings
        self.textbox.insert(tk.END, "\n COLLECTED RESEARCH GAPS\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        for gap in summary_result.get("collected_gaps", []):
            self.textbox.insert(tk.END, f"• {gap}\n")
        
        self.textbox.insert(tk.END, "\n COLLECTED METHODS\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        for method in summary_result.get("collected_methods", []):
            self.textbox.insert(tk.END, f"• {method}\n")
        
        self.textbox.insert(tk.END, "\n COLLECTED BASELINES\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        for baseline in summary_result.get("collected_baselines", []):
            self.textbox.insert(tk.END, f"• {baseline}\n")
        
        self.textbox.insert(tk.END, "\n COLLECTED DATASETS\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        for dataset in summary_result.get("collected_datasets", []):
            self.textbox.insert(tk.END, f"• {dataset}\n")
        
        self.textbox.insert(tk.END, "\n PROPOSED METHODOLOGY\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n\n")
        self.textbox.insert(tk.END, summary_result.get("proposed_methodology", "N/A") + "\n")

    def _show_memory_stats(self):
        stats = self.memory_agent.get_memory_stats()
        self.textbox.delete(1.0, tk.END)
        
        self.textbox.insert(tk.END, "=" * 100 + "\n")
        self.textbox.insert(tk.END, "MEMORY STATISTICS\n")
        self.textbox.insert(tk.END, "=" * 100 + "\n\n")
        
        self.textbox.insert(tk.END, f"Total Sessions: {stats['total_sessions']}\n")
        self.textbox.insert(tk.END, f"Total Papers Stored: {stats['total_papers']}\n")
        self.textbox.insert(tk.END, f"Total Analyses: {stats['total_analyses']}\n\n")
        
        if stats['latest_session']:
            self.textbox.insert(tk.END, "Latest Session:\n")
            self.textbox.insert(tk.END, f"  Session ID: {stats['latest_session']['session_id']}\n")
            self.textbox.insert(tk.END, f"  Query: {stats['latest_session']['query']}\n")
            self.textbox.insert(tk.END, f"  Time: {stats['latest_session']['timestamp']}\n")
            self.textbox.insert(tk.END, f"  Papers Found: {stats['latest_session']['paper_count']}\n")
        
        self.status_var.set("Memory statistics displayed")

    def _display_papers(self, papers):
        for i, p in enumerate(papers, 1):
            self.textbox.insert(tk.END, f"\n[{i}] {p.get('title', 'N/A')}\n")
            self.textbox.insert(tk.END, f"    Year: {p.get('publication_year', 'N/A')} | ")
            self.textbox.insert(tk.END, f"Citations: {p.get('citation_count', 0)}\n")
            
            # Authors - handle None
            authors = p.get('authors') or []
            author_str = ', '.join(authors[:3]) if authors else 'N/A'
            self.textbox.insert(tk.END, f"    Authors: {author_str}\n")
            
            self.textbox.insert(tk.END, f"    URL: {p.get('url', 'N/A')}\n")
            self.textbox.insert(tk.END, f"    PDF: {p.get('pdf_url', 'N/A')}\n")
            self.textbox.insert(tk.END, f"    DOI: {p.get('doi', 'N/A')}\n")
            
            # Keywords - handle None
            keywords_list = p.get('keywords') or []
            keywords = ', '.join(keywords_list[:5]) if keywords_list else ''
            if keywords:
                self.textbox.insert(tk.END, f"    Keywords: {keywords}\n")
            
            # TL;DR
            if p.get('tldr'):
                self.textbox.insert(tk.END, f"    TL;DR: {p.get('tldr')}\n")
            
            # Abstract
            abstract = p.get('abstract') or ''
            if abstract:
                abstract_display = abstract[:400] + ('...' if len(abstract) > 400 else '')
                self.textbox.insert(tk.END, f"    Abstract: {abstract_display}\n")
            
            self.textbox.insert(tk.END, "    " + "-" * 90 + "\n")

    def _display_datasets(self, datasets):
        for i, d in enumerate(datasets, 1):
            self.textbox.insert(tk.END, f"\n[{i}] {d['title']}\n")
            self.textbox.insert(tk.END, f"    Source: {d['source']}\n")
            self.textbox.insert(tk.END, f"    URL: {d['url']}\n")
            self.textbox.insert(tk.END, f"    Description: {d['snippet']}\n")
            self.textbox.insert(tk.END, "    " + "-" * 90 + "\n")


# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    print("Starting AI Research Assistant...")
    
    # Initialize agents
    retrieval_agent = RetrievalAgent(
        semantic_api_key=SEMANTIC_SCHOLAR_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        max_results=20
    )
    
    memory_agent = MemoryAgent(
        memory_file='research_memory.json',
        analysis_file='research_analysis.json'
    )
    
    print("Agents initialized successfully")
    print("Launching UI...")
    
    # Create and run UI
    root = tk.Tk()
    app = ResearchAssistantUI(root, retrieval_agent, memory_agent)
    root.mainloop()