from runner import run_jobs
jobs = []
for h in [0, 1, 2]:
    jobs.append(dict(tag=f"n3 s1 hidden={h} plain", community=dict(n_species=3, seed=1), train=dict(hidden=h)))
    jobs.append(dict(tag=f"n3 s1 hidden={h} ORACLE warm-up", community=dict(n_species=3, seed=1), train=dict(hidden=h, oracle_warmup=True)))
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp02.jsonl')
