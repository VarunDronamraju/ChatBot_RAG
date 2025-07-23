# app/utils/logger.py

import json
import os
from datetime import datetime

LOG_FILE = "app/utils/eval_logs.json"

def log_eval(question, answer, source, response_time, citations):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "source": source,
        "response_time": round(response_time, 2),
        "citations": citations,
    }

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([log_entry], f, indent=2)
    else:
        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append(log_entry)
            f.seek(0)
            json.dump(logs, f, indent=2)
