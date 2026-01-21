
import json
import requests
import sys
import time
from pathlib import Path

# Load length payload
# Try to find a recent large request log
logs_dir = Path("logs")
log_files = sorted(logs_dir.glob("**/2_req_model.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not log_files:
    print("No log files found")
    sys.exit(1)

log_file = log_files[0]
print(f"Using log file: {log_file}")

with open(log_file, "r") as f:
    data = json.load(f)

if "model" not in data:
    data["model"] = "gpt-4o"

data["stream"] = True

print(f"Sending request with {len(data['messages'])} messages...")
text_len = sum(len(m.get('content', '')) for m in data['messages'])
print(f"Approx payload chars: {text_len}")

start_time = time.time()
try:
    with requests.post("http://127.0.0.1:8000/v1/chat/completions", json=data, stream=True) as r:
        r.raise_for_status()
        print("Response stream started...")
        first_byte = time.time()
        print(f"TTFT (Time To First Token): {first_byte - start_time:.3f}s")
        
        chunk_count = 0
        for line in r.iter_lines():
            if line:
                chunk_count += 1
                if chunk_count % 10 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
        
        end_time = time.time()
        print(f"\nTotal time: {end_time - start_time:.3f}s")

except Exception as e:
    print(f"Request failed: {e}")
