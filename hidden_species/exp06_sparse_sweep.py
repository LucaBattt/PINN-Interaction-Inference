"""Noise-free sweep: one species has only m data points per trajectory (m = 1 means only its initial abundance)."""
import hs_pinn as hs
from runner import run_jobs
SEEDS = {3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 13, 18]}   # first three random communities that stay bounded with all species present
jobs = []
for n in [3, 4, 5]:
    for s in SEEDS[n]:
        com = hs.make_community(n, seed=s)
        h = hs.most_influential_species(com)
        for m in [1, 3, 5, 10]:
            jobs.append(dict(tag=f"n{n} s{s} h={h} {m:2d}pts", community=dict(n_species=n, seed=s),
                             train=dict(hidden=h, n_hidden_pts=m, lbfgs_iters=5000)))
jobs.sort(key=lambda j: -j['community']['n_species'])   # long jobs first
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp06.jsonl')
