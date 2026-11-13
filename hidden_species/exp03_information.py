from runner import run_jobs
jobs = []
for h in [0, 1, 2]:
    base = dict(n_species=3, seed=1)
    jobs.append(dict(tag=f"n3 s1 h={h} fully hidden, r known", community=base, train=dict(hidden=h, r_known=True)))
    jobs.append(dict(tag=f"n3 s1 h={h} fully hidden, r+diag known", community=base, train=dict(hidden=h, r_known=True, diag_known=True)))
    for m in [1, 3, 5, 10]:
        jobs.append(dict(tag=f"n3 s1 h={h} {m:2d} pts", community=base, train=dict(hidden=h, n_hidden_pts=m)))
    for m in [1, 3]:
        jobs.append(dict(tag=f"n3 s1 h={h} {m:2d} pts, r known", community=base, train=dict(hidden=h, n_hidden_pts=m, r_known=True)))
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp03.jsonl')
