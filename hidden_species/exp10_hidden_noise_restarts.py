"""Fully hidden species: (a) known growth rates with noisy data, (b) restarts of the runs that failed in exp07."""
import hs_pinn as hs
from runner import run_jobs
SEEDS = {3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 13, 18]}
FAILED_NOTHING_KNOWN = [(3, 1), (4, 0), (4, 1), (4, 4), (5, 0), (5, 13), (5, 18)]
jobs = []
for n in [3, 4, 5]:
    for s in SEEDS[n]:
        h = hs.most_influential_species(hs.make_community(n, seed=s))
        for noise in [0.01, 0.03]:
            jobs.append(dict(tag=f"n{n} s{s} h={h} hidden | r known | noise={noise}", variant='r known', community=dict(n_species=n, seed=s),
                             train=dict(hidden=h, r_known=True, noise_std=noise, lbfgs_iters=3000)))
        if (n, s) in FAILED_NOTHING_KNOWN:
            for init in [1, 2]:
                jobs.append(dict(tag=f"n{n} s{s} h={h} hidden | nothing known | restart {init}", variant='nothing known', community=dict(n_species=n, seed=s),
                                 train=dict(hidden=h, seed=init, lbfgs_iters=5000)))
        if (n, s) == (5, 18):
            for init in [1, 2]:
                jobs.append(dict(tag=f"n{n} s{s} h={h} hidden | r known | restart {init}", variant='r known', community=dict(n_species=n, seed=s),
                                 train=dict(hidden=h, r_known=True, seed=init, lbfgs_iters=5000)))
jobs.sort(key=lambda j: -j['community']['n_species'])
if __name__ == '__main__':
    print(len(jobs), 'jobs')
    run_jobs(jobs, 'results/exp10.jsonl')
