import json
import os
from datetime import datetime


class MemoryAgent:
    def __init__(self, memory_file='research_memory.json', analysis_file='research_analysis.json'):
        self.memory_file = memory_file
        self.analysis_file = analysis_file
        self.memory = self._load_memory()
        self.analyses = self._load_analyses()
    
    def _load_memory(self):
        """Load existing memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure all required keys exist
                    if 'sessions' not in data:
                        data['sessions'] = []
                    if 'papers' not in data:
                        data['papers'] = {}
                    print(f" Loaded memory from {self.memory_file}")
                    return data
            except json.JSONDecodeError:
                print(" Memory file corrupted, creating new memory")
                return {'sessions': [], 'papers': {}}
        print(f" Creating new memory file: {self.memory_file}")
        return {'sessions': [], 'papers': {}}
    
    def _load_analyses(self):
        """Load existing analyses from separate file"""
        if os.path.exists(self.analysis_file):
            try:
                with open(self.analysis_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f" Loaded analyses from {self.analysis_file}")
                    print(f" Found {len(data)} existing analyses")
                    return data
            except json.JSONDecodeError:
                print(" Analysis file corrupted, creating new file")
                return {}
        print(f" Creating new analysis file: {self.analysis_file}")
        return {}
    
    def store_papers(self, session_id, query, papers_metadata):
        """Store retrieved papers in memory"""
        session_data = {
            'session_id': session_id,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'paper_count': len(papers_metadata),
            'paper_ids': [p['paper_id'] for p in papers_metadata]
        }
        
        # Store session
        self.memory['sessions'].append(session_data)
        
        # Store individual papers (avoid duplicates)
        for paper in papers_metadata:
            paper_id = paper['paper_id']
            if paper_id not in self.memory['papers']:
                self.memory['papers'][paper_id] = paper
        
        self._save_memory()
        print(f" Stored {len(papers_metadata)} papers in memory")
        print(f" Session ID: {session_id}")
        return session_id
    
    def store_analysis(self, session_id, analysis):
        """Store analysis results from summarizer agent in separate file"""
        print(f"\n{'='*80}")
        print(f" Storing analysis for session: {session_id}")
        
        self.analyses[session_id] = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'analysis': analysis
        }
        
        print(f" Analysis data keys: {analysis.keys()}")
        print(f" Per-paper summaries: {len(analysis.get('per_paper', []))}")
        print(f" Research gaps: {len(analysis.get('collected_gaps', []))}")
        print(f" Methods: {len(analysis.get('collected_methods', []))}")
        print(f" Baselines: {len(analysis.get('collected_baselines', []))}")
        print(f" Datasets: {len(analysis.get('collected_datasets', []))}")
        
        self._save_analyses()
        
        print(f" Successfully stored analysis for session {session_id}")
        print(f" Total analyses in file: {len(self.analyses)}")
        print(f"{'='*80}\n")
    
    def store_evaluation(self, session_id, evaluation_report):
        """Store evaluation results in the analysis file"""
        # Store under the session's analysis
        if session_id in self.analyses:
            self.analyses[session_id]['evaluation'] = evaluation_report
        else:
            self.analyses[session_id] = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'evaluation': evaluation_report
            }
        
        self._save_analyses()
        print(f" Stored evaluation for session {session_id}")
    
    def get_papers_by_session(self, session_id):
        """Retrieve papers from a specific session"""
        for session in self.memory['sessions']:
            if session['session_id'] == session_id:
                paper_ids = session['paper_ids']
                papers = [self.memory['papers'][pid] for pid in paper_ids if pid in self.memory['papers']]
                print(f"📚 Retrieved {len(papers)} papers for session {session_id}")
                return papers
        print(f" No papers found for session {session_id}")
        return []
    
    def get_all_papers(self):
        """Get all stored papers"""
        return list(self.memory['papers'].values())
    
    def get_analysis(self, session_id):
        """Retrieve analysis for a session from separate file"""
        analysis = self.analyses.get(session_id)
        if analysis:
            print(f" Found analysis for session {session_id}")
        else:
            print(f" No analysis found for session {session_id}")
            print(f" Available sessions with analysis: {list(self.analyses.keys())}")
        return analysis
    
    def get_evaluation(self, session_id):
        """Retrieve evaluation for a session"""
        if session_id in self.analyses:
            evaluation = self.analyses[session_id].get('evaluation')
            if evaluation:
                print(f" Found evaluation for session {session_id}")
            else:
                print(f" No evaluation found for session {session_id}")
            return evaluation
        print(f" No data found for session {session_id}")
        return None
    
    def get_session_query(self, session_id):
        """Get the query for a specific session"""
        for session in self.memory['sessions']:
            if session['session_id'] == session_id:
                return session['query']
        return None
    
    def search_memory(self, keyword):
        """Search papers by keyword in title or abstract"""
        results = []
        keyword_lower = keyword.lower()
        
        for paper in self.memory['papers'].values():
            if (keyword_lower in paper.get('title', '').lower() or 
                keyword_lower in paper.get('abstract', '').lower()):
                results.append(paper)
        
        print(f"🔍 Found {len(results)} papers matching '{keyword}'")
        return results
    
    def get_memory_stats(self):
        """Get statistics about stored data"""
        stats = {
            'total_sessions': len(self.memory.get('sessions', [])),
            'total_papers': len(self.memory.get('papers', {})),
            'total_analyses': len(self.analyses),
            'latest_session': self.memory['sessions'][-1] if self.memory.get('sessions') else None
        }
        return stats
    
    def _save_memory(self):
        """Save memory to file with error handling"""
        try:
            temp_file = self.memory_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  
 
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
            os.rename(temp_file, self.memory_file)
            
            # Verify file was written
            file_size = os.path.getsize(self.memory_file)
            print(f"💾 Memory saved to {self.memory_file} ({file_size} bytes)")
            
        except Exception as e:
            print(f" ERROR saving memory: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_analyses(self):
        """Save analyses to separate file with error handling"""
        try:
            print(f"💾 Saving analyses to {self.analysis_file}...")
            
            temp_file = self.analysis_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.analyses, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno()) 
            
            if os.path.exists(self.analysis_file):
                os.remove(self.analysis_file)
            os.rename(temp_file, self.analysis_file)
            
            # Verify file was written
            file_size = os.path.getsize(self.analysis_file)
            print(f"✅ Analyses saved to {self.analysis_file} ({file_size} bytes)")
            
            # Double-check by reading it back
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                verification = json.load(f)
                print(f"✅ Verification: File contains {len(verification)} analyses")
            
        except Exception as e:
            print(f" ERROR saving analyses: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_memory(self):
        """Clear all stored data (use with caution)"""
        self.memory = {'sessions': [], 'papers': {}}
        self.analyses = {}
        self._save_memory()
        self._save_analyses()
        print(" Memory and analyses cleared")
    
    def list_all_sessions(self):
        """List all sessions with their basic info"""
        print("\n" + "="*80)
        print("ALL SESSIONS")
        print("="*80)
        
        for idx, session in enumerate(self.memory.get('sessions', []), 1):
            session_id = session['session_id']
            print(f"\n[{idx}] Session ID: {session_id}")
            print(f"    Query: {session['query']}")
            print(f"    Timestamp: {session['timestamp']}")
            print(f"    Papers: {session['paper_count']}")
            
            # Check if analysis exists
            has_analysis = session_id in self.analyses
            has_evaluation = False
            if has_analysis and 'evaluation' in self.analyses[session_id]:
                has_evaluation = True
            
            print(f"    Analysis: {' Yes' if has_analysis else ' No'}")
            print(f"    Evaluation: {' Yes' if has_evaluation else ' No'}")
        
        print("\n" + "="*80 + "\n")
    
    def export_analysis_to_file(self, session_id, output_file=None):
        """Export a specific analysis to a text file"""
        analysis = self.get_analysis(session_id)
        
        if not analysis:
            print(f" No analysis found for session {session_id}")
            return False
        
        if output_file is None:
            output_file = f"analysis_{session_id}.txt"
        
        try:
            analysis_data = analysis.get('analysis', {})
            evaluation_data = analysis.get('evaluation')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("RESEARCH ANALYSIS REPORT\n")
                f.write("=" * 100 + "\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Timestamp: {analysis.get('timestamp', 'N/A')}\n")
                f.write("=" * 100 + "\n\n")
                
                # Evaluation scores if available
                if evaluation_data:
                    f.write("EVALUATION SCORES\n")
                    f.write("=" * 100 + "\n")
                    agg = evaluation_data.get('aggregate_scores', {})
                    f.write(f"Overall Status: {' PASSED' if evaluation_data.get('is_valid') else '❌ FAILED'}\n")
                    f.write(f"Average Relevance: {agg.get('avg_relevance', 'N/A')}/5.0\n")
                    f.write(f"Average Completeness: {agg.get('avg_completeness', 'N/A')}/5.0\n")
                    f.write(f"Average Novelty: {agg.get('avg_novelty', 'N/A')}/5.0\n\n")
                
                # Per-paper summaries
                f.write("PER-PAPER SUMMARIES\n")
                f.write("=" * 100 + "\n\n")
                
                for idx, summary in enumerate(analysis_data.get("per_paper", []), 1):
                    f.write(f"[Paper {idx}]\n")
                    f.write(f"Summary: {summary.get('summary', 'N/A')}\n")
                    
                    if summary.get('methods'):
                        f.write(f"Methods: {', '.join(summary['methods'])}\n")
                    if summary.get('baselines'):
                        f.write(f"Baselines: {', '.join(summary['baselines'])}\n")
                    if summary.get('datasets'):
                        f.write(f"Datasets: {', '.join(summary['datasets'])}\n")
                    if summary.get('gaps'):
                        f.write(f"Research Gaps: {', '.join(summary['gaps'])}\n")
                    if summary.get('error'):
                        f.write(f"Error: {summary['error']}\n")
                    
                    f.write("-" * 90 + "\n\n")
                
                # Collected data
                f.write("\nCOLLECTED RESEARCH GAPS\n")
                f.write("=" * 100 + "\n")
                for gap in analysis_data.get("collected_gaps", []):
                    f.write(f"• {gap}\n")
                
                f.write("\nCOLLECTED METHODS\n")
                f.write("=" * 100 + "\n")
                for method in analysis_data.get("collected_methods", []):
                    f.write(f"• {method}\n")
                
                f.write("\nCOLLECTED BASELINES\n")
                f.write("=" * 100 + "\n")
                for baseline in analysis_data.get("collected_baselines", []):
                    f.write(f"• {baseline}\n")
                
                f.write("\nCOLLECTED DATASETS\n")
                f.write("=" * 100 + "\n")
                for dataset in analysis_data.get("collected_datasets", []):
                    f.write(f"• {dataset}\n")
                
                f.write("\nPROPOSED METHODOLOGY\n")
                f.write("=" * 100 + "\n\n")
                f.write(analysis_data.get("proposed_methodology", "N/A") + "\n")
            
            print(f" Analysis exported to {output_file}")
            return True
            
        except Exception as e:
            print(f" Error exporting analysis: {e}")
            return False