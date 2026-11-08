import requests, time

def generate(prompt, model="llama3.2:3b"):
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=120)
    r.raise_for_status()
    return r.json()

t0 = time.perf_counter()
res = generate("In one sentence, what is data leakage in ML?")
print(res["response"])
print(f"wall: {time.perf_counter() - t0:.2f}s")
print("tokens/s:", res["eval_count"] / (res["eval_duration"] / 1e9))
