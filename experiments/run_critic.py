import os,sys,json,time,subprocess
from concurrent.futures import ThreadPoolExecutor
PY=sys.executable; REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT=os.path.join(REPO,"src","critic_ablation.py")
jobs=json.load(open(sys.argv[1])); mp=int(sys.argv[2]) if len(sys.argv)>2 else 8
def run(j,i):
    time.sleep(i*1.5); args=[PY,SCRIPT]
    for k,v in j.items(): args+=[f"--{k}",str(v)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    ld=os.path.join(REPO,"results","logs"); os.makedirs(ld,exist_ok=True)
    with open(os.path.join(ld,j["tag"]+".log"),"w") as lf:
        return j["tag"], subprocess.run(args,stdout=lf,stderr=subprocess.STDOUT,env=env,cwd=REPO).returncode
print(f"[critic] {len(jobs)} jobs mp={mp}",flush=True); t0=time.time()
with ThreadPoolExecutor(max_workers=mp) as ex:
    for f in [ex.submit(run,j,i%mp) for i,j in enumerate(jobs)]:
        tag,rc=f.result(); print(f"[done] {tag} rc={rc} ({time.time()-t0:.0f}s)",flush=True)
print(f"[critic] all done in {time.time()-t0:.0f}s",flush=True)
