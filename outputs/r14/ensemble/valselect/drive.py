"""Tiny job driver: runs `python3 <script> <args...>` jobs with bounded parallelism.
Skips jobs whose output file already exists. Usage: drive.py <jobs.txt> <nproc>"""
import os, subprocess, sys, threading, queue, time
D = "outputs/r14/ensemble/valselect"
jobs = [l.split() for l in open(sys.argv[1]) if l.strip()]
N = int(sys.argv[2])
q = queue.Queue()
for j in jobs: q.put(j)
lock = threading.Lock()
def worker():
    while True:
        try: j = q.get_nowait()
        except queue.Empty: return
        script, args = j[0], j[1:]
        tag = "_".join(args)
        out = f"{D}/preds/{args[0]}/{args[1]}__seed{args[2]}.json"
        lk = out + ".lock"
        if os.path.exists(out):
            with lock: print("SKIP", tag, flush=True); q.task_done(); continue
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            with lock: print("LOCKED", tag, flush=True); q.task_done(); continue
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        with open(f"{D}/logs/{script}_{tag}.log", "a") as lg:
            rc = subprocess.call(["nice", "-n", "15", "python3", f"{D}/{script}.py", *args],
                                 stdout=lg, stderr=subprocess.STDOUT, env=env)
        try: os.remove(lk)
        except OSError: pass
        with lock: print("OK" if rc == 0 else f"FAIL({rc})", tag, f"{time.time()-t0:.0f}s", flush=True)
        q.task_done()
ts = [threading.Thread(target=worker, daemon=True) for _ in range(N)]
[t.start() for t in ts]; [t.join() for t in ts]
print("QUEUE DONE", flush=True)
