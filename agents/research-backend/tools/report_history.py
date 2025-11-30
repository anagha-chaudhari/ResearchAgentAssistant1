import json
import os

hist_file = "backend/data/report_history.json"
os.makedirs(os.path.dirname(hist_file),exist_ok=True)

def save_report(topic,markdown):
    if os.path.exists(hist_file):
        with open(hist_file,"r",encoding="utf-8") as f:
            data=json.load(f)
            
    else:
        data = []
        
    data.insert(0, {"topic": topic, "markdown":markdown})
    
    with open (hist_file,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)
        
def load_reports():
    if not os.path.exists(hist_file):
        return []
    with open(hist_file,"r",encoding="utf-8") as f:
        return json.load(f)