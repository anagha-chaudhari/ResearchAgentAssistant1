import requests
from datetime import datetime
import time


class RetrievalAgent:
    def __init__(self, semantic_api_key, google_api_key=None, google_cse_id=None, max_results=25):
        self.semantic_api_key = semantic_api_key
        self.semantic_api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.max_results = max_results
        
        # Google Custom Search API (optional for dataset search)
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        
        # Store retrieved papers for access by other agents
        self.retrieved_papers = []
        self.retrieved_datasets = []

    def search_recent_papers(self, query, years_back=3, min_results=5,exclude_ids=None):
        """Search for recent papers with full metadata extraction"""
        print(f" Searching for papers on: '{query}'")
        
        current_year = datetime.now().year
        best_papers = []
        year_window = years_back
        
        while True:
            from_year = current_year - year_window + 1
            papers = self._get_papers(query, from_year)
            filtered = self._filter_by_methodology(papers)
            
            if len(filtered) >= min_results or year_window >= 8:
                best_papers = filtered
                break
            year_window = 8
        
        sorted_papers = sorted(
            best_papers, 
            key=lambda x: (x.get('citationCount', 0), x.get('year', 0)), 
            reverse=True
        )[:10]
        
        # Extract complete metadata
        structured_papers = self._extract_metadata(sorted_papers)
        if exclude_ids is not None:
            structured_papers = [p for p in structured_papers if p['paper_id'] not in exclude_ids]
        self.retrieved_papers = structured_papers
        
        print(f" Retrieved {len(structured_papers)} papers")
        return structured_papers
    

    def search_datasets(self, query, num_results=5):
        """Search for relevant datasets using Google Custom Search"""
        if not self.google_api_key or not self.google_cse_id:
            print(" Google API credentials not configured")
            return []
        
        print(f" Searching for datasets on: '{query}'")
        dataset_query = f"{query} dataset kaggle github"
        
        params = {
            'key': self.google_api_key,
            'cx': self.google_cse_id,
            'q': dataset_query,
            'num': num_results
        }
        
        try:
            response = requests.get(self.google_search_url, params=params)
            if response.status_code != 200:
                print(f" Google Search API Error: {response.status_code}")
                return []
            
            data = response.json()
            datasets = []
            
            for item in data.get('items', []):
                datasets.append({
                    'title': item.get('title', 'N/A'),
                    'url': item.get('link', 'N/A'),
                    'snippet': item.get('snippet', ''),
                    'source': self._identify_source(item.get('link', ''))
                })
            
            self.retrieved_datasets = datasets
            print(f" Retrieved {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            print(f" Error searching datasets: {e}")
            return []

    def _get_papers(self, topic, from_year):
        """Get papers from Semantic Scholar API with extended metadata"""
        papers, offset = [], 0
        headers = {"x-api-key": self.semantic_api_key}
        
        for _ in range(2):
            params = {
                "query": topic,
                "fields": "paperId,title,url,abstract,authors,year,citationCount,tldr,externalIds,publicationTypes,publicationDate,journal,fieldsOfStudy",
                "limit": self.max_results,
                "offset": offset,
                "year": f"{from_year}-{datetime.now().year}"
            }
            
            response = requests.get(self.semantic_api_url, params=params, headers=headers)
            time.sleep(1)  
            
            if response.status_code != 200:
                print(f" Semantic Scholar API Error: {response.status_code}")
                break
            
            data = response.json()
            papers.extend(data.get("data", []))
            offset += self.max_results
            
            if len(data.get("data", [])) < self.max_results:
                break
        
        filtered_papers = [p for p in papers if p.get("year") and p.get("year") >= from_year]
        return filtered_papers

    def _filter_by_methodology(self, papers):
        """Filter papers by methodology keywords"""
        keywords = ['method', 'approach', 'technique', 'propose', 'introduce', 'novel', 'architecture']
        
        def methodology_score(paper):
            abstract = paper.get('abstract') or ''
            tldr_text = ''
            if paper.get('tldr') and paper.get('tldr').get('text'):
                tldr_text = paper['tldr']['text']
            
            text = (abstract + ' ' + tldr_text).lower()
            return sum(kw in text for kw in keywords)
        
        scored = [(methodology_score(p), p) for p in papers]
        high_score = [p for s, p in scored if s > 0]
        
        return high_score if high_score else [p for s, p in scored]

    def _extract_metadata(self, papers):
        """Extract comprehensive metadata for memory storage"""
        structured_papers = []
        for paper in papers:

            metadata = {
                'paper_id': paper.get('paperId', 'N/A'), 
                'title': paper.get('title', 'N/A'),
                'url': paper.get('url', 'N/A'),
 
                'authors': [a.get('name') for a in paper.get('authors', [])],

                'publication_year': paper.get('year', 'N/A'),
                'publication_date': paper.get('publicationDate', 'N/A'),
                'journal': paper.get('journal', {}).get('name', 'N/A') if paper.get('journal') else 'N/A',

                'doi': paper.get('externalIds', {}).get('DOI', 'N/A') if paper.get('externalIds') else 'N/A',
                'arxiv_id': paper.get('externalIds', {}).get('ArXiv', 'N/A') if paper.get('externalIds') else 'N/A',

                'keywords': paper.get('fieldsOfStudy', []),

                'abstract': paper.get('abstract', ''),
                'tldr': paper.get('tldr', {}).get('text', '') if paper.get('tldr') else '',

                'citation_count': paper.get('citationCount', 0),
                'publication_types': paper.get('publicationTypes', []),

                'pdf_url': self._construct_pdf_url(paper),  # <--- use paper

                'retrieved_at': datetime.now().isoformat()
          }
            structured_papers.append(metadata)
        
        return structured_papers

    def _construct_pdf_url(self, paper):
        """Construct PDF URL from available identifiers"""
        external_ids = paper.get('externalIds', {})
        
        # Try ArXiv first
        if external_ids.get('ArXiv'):
            return f"https://arxiv.org/pdf/{external_ids['ArXiv']}.pdf"
        
        # Try DOI
        if external_ids.get('DOI'):
            return f"https://doi.org/{external_ids['DOI']}"
        
        return 'N/A'

    def _identify_source(self, url):
        """Identify dataset source from URL"""
        sources = {
            'kaggle.com': 'Kaggle',
            'github.com': 'GitHub',
            'huggingface.co': 'Hugging Face',
            'uci.edu': 'UCI ML Repository',
            'openml.org': 'OpenML',
            'zenodo.org': 'Zenodo',
            'data.gov': 'Data.gov',
            'data.gov.in': 'Indian Open Data'
        }
        
        for key, name in sources.items():
            if key in url.lower():
                return name
        return 'Other'