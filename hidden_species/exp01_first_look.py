from runner import run_jobs
jobs = []
for h in [0, 1, 2]:
    for r_known in [False, True]:
        jobs.append(dict(tag=f"n3 s1 hidden={h} r_known={r_known}", community=dict(n_species=3, seed=1),
                         train=dict(hidden=h, r_known=r_known)))
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp01.jsonl')
