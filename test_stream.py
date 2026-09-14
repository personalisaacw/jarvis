import requests
import time
import json

URL = "http://127.0.0.1:11435/api/chat"

payload = {
    "model": "qwen3-4b-fast",
    "messages": [
        {
            "role": "user",
            "content": "Give me an explanation of hashmaps in 2 sentences."
        }
    ],
    "think": True,
    "stream": True
}

start = time.perf_counter()
first_chunk_time = None
chunks = 0
tokens_text = ""

print("Sending request to Ollama...\n")

with requests.post(URL, json=payload, stream=True) as response:

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        now = time.perf_counter()

        if first_chunk_time is None:
            first_chunk_time = now

            print(
                f"\nFIRST CHUNK RECEIVED"
            )

            print(
                f"TTFT: {first_chunk_time - start:.3f} seconds\n"
            )

        chunks += 1

        data = json.loads(line)

        text = data.get("message", {}).get("content", "")

        if text:
            tokens_text += text
            print(
                f"[{now - start:7.3f}s] {text}",
                end="",
                flush=True
            )

end = time.perf_counter()

print("\n")
print("========== RESULTS ==========")
print(f"TTFT:           {first_chunk_time - start:.3f}s")
print(f"Total time:     {end - start:.3f}s")
print(f"Chunks:         {chunks}")
print(f"Generation:     {(end - first_chunk_time):.3f}s")