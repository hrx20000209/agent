import fcntl,os,sys,runpy
from pathlib import Path
for key in ['TPL_DEMOS','EXPLORE_FIT','WARM_EXPLORE','NO_EXEC','NO_SYS','PIPE_ONLY','GRAPH_PROMPT','RAND_EXPLORE','SAME_TPL','AGGR_COMMIT','NAV_UNAMBIG','MAI_SHARED_STATE','MAI_EPISODE_ROOTS_JSON','MAI_SMOKE_EMPTY_STATS']:os.environ.pop(key,None)
f=open('/tmp/mai_device_experiment.claim.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=Path('/androidworld/experiments/trace_reachability_validation/scripts/owl_upstream_probe.py');sys.path.insert(0,str(p.parent));sys.argv[0]=str(p);runpy.run_path(str(p),run_name='__main__')
