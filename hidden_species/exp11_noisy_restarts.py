"""Restarts of the three noisy 'fully hidden, r known' runs whose fit to the data was clearly worse than the noise level."""
import hs_pinn as hs
from runner import run_jobs
jobs = []
for (n, s, noise) in [(5, 18, 0.01), (5, 13, 0.03), (5, 18, 0.03)]:
    h = hs.most_influential_species(hs.make_community(n, seed=s))
    for init in [1, 2]:
        jobs.append(dict(tag=f"n{n} s{s} h={h} hidden | r known | noise={noise} | restart {init}", variant='r known', community=dict(n_species=n, seed=s),
                         train=dict(hidden=h, r_known=True, noise_std=noise, seed=init, lbfgs_iters=3000)))
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp11.jsonl')
