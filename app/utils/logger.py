# app/utils/logger.py

import os
import json
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "eval_logs.json")

def log_eval(question, answer, source_type, response_time, citations):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "source": source_type,
        "response_time_secs": round(response_time, 2),
        "citations": citations,
    }

    # Create log file if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([log_entry], f, indent=2)
    else:
        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=2)
