import json
import os

HISTORY_FILE = "backend/data/report_history.json"
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


def save_report(topic: str, markdown: str):
    # Load existing history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Insert newest at top
    data.insert(0, {
        "topic": topic,
        "markdown": markdown
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_reports(limit: int = 3):
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        reports = json.load(f)

    seen_topics = set()
    unique_latest = []

    # Already newest-first
    for r in reports:
        topic = r.get("topic")
        if topic and topic not in seen_topics:
            unique_latest.append(r)
            seen_topics.add(topic)

        if len(unique_latest) >= limit:
            break

    return unique_latest