"""Hard case: the sparse species is the one with the weakest effect on the others (noise-free and noisy)."""
import hs_pinn as hs
from runner import run_jobs
SEEDS = {3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 13, 18]}
jobs = []
for n in [3, 4, 5]:
    for s in SEEDS[n]:
        h = hs.least_influential_species(hs.make_community(n, seed=s))
        for m in [1, 3, 10]:
            jobs.append(dict(tag=f"n{n} s{s} h={h}(least) {m:2d}pts", community=dict(n_species=n, seed=s),
                             train=dict(hidden=h, n_hidden_pts=m, lbfgs_iters=5000)))
        jobs.append(dict(tag=f"n{n} s{s} h={h}(least) 10pts noise=0.01", community=dict(n_species=n, seed=s),
                         train=dict(hidden=h, n_hidden_pts=10, noise_std=0.01, lbfgs_iters=3000)))
jobs.sort(key=lambda j: -j['community']['n_species'])
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp09.jsonl')
