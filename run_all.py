import subprocess, sys

steps = ["train.py", "predict.py", "local_sentiment.py"]
for s in steps:
    print(f"\n=== {s} ===")
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
       print(f"{s} failed (exit {r.returncode}), stopping.")
       break
