import time
from transformers import pipeline

t0 = time.perf_counter()
clf = pipeline("sentiment-analysis",
              model="distilbert-base-uncased-finetuned-sst-2-english")
print(f"load: {time.perf_counter() - t0:.2f}s")

texts = ["This deployment went smoothly.", "The API keeps timing out."] * 50

clf(texts[:2]) #warmup
t0 = time.perf_counter()
out = clf(texts, batch_size=16)
dt = time.perf_counter() -t0
print(f"{len(texts)/dt:.1f} req/s, {1000*dt/len(texts):.1f} ms/req")
print(out[:2])
