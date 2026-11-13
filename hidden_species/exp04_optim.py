from runner import run_jobs
jobs = []
base = dict(n_species=3, seed=1)
cases = {'h=1 3pts': dict(hidden=1, n_hidden_pts=3), 'h=0 1pt': dict(hidden=0, n_hidden_pts=1),
         'h=1 hidden r known': dict(hidden=1, r_known=True), 'h=2 hidden': dict(hidden=2)}
variants = {'varpro': dict(varpro=True),
            'lbfgs': dict(lbfgs_iters=3000, dtype='float64'),
            'varpro+lbfgs': dict(varpro=True, lbfgs_iters=3000, dtype='float64')}
for cn, c in cases.items():
    for vn, v in variants.items():
        jobs.append(dict(tag=f"n3 s1 {cn} | {vn}", community=base, train={**c, **v}))
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp04.jsonl')
