"""Noisy data: sparse species with 3 or 10 points, plus the fully observed PINN as reference."""
import sys
import hs_pinn as hs
from runner import run_jobs
SEEDS = {3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 13, 18]}
NOISES = [float(a) for a in sys.argv[1:]] or [0.01, 0.03]
jobs = []
for n in [3, 4, 5]:
    for s in SEEDS[n]:
        h = hs.most_influential_species(hs.make_community(n, seed=s))
        for noise in NOISES:
            for m in [3, 10]:
                jobs.append(dict(tag=f"n{n} s{s} h={h} {m:2d}pts noise={noise}", community=dict(n_species=n, seed=s),
                                 train=dict(hidden=h, n_hidden_pts=m, noise_std=noise, lbfgs_iters=3000)))
            jobs.append(dict(tag=f"n{n} s{s} fully observed noise={noise}", community=dict(n_species=n, seed=s),
                             train=dict(noise_std=noise, lbfgs_iters=3000)))
jobs.sort(key=lambda j: -j['community']['n_species'])
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp08.jsonl')
