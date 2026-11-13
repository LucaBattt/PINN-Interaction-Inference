"""Noise-free sweep: one species has no data at all. What has to be known for the inference to work?"""
import hs_pinn as hs
from runner import run_jobs
SEEDS = {3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 13, 18]}
VARIANTS = {'nothing known': dict(), 'r known': dict(r_known=True), 'r + self-regulation known': dict(r_known=True, diag_known=True)}
jobs = []
for n in [3, 4, 5]:
    for s in SEEDS[n]:
        h = hs.most_influential_species(hs.make_community(n, seed=s))
        for name, v in VARIANTS.items():
            jobs.append(dict(tag=f"n{n} s{s} h={h} hidden | {name}", variant=name, community=dict(n_species=n, seed=s),
                             train=dict(hidden=h, lbfgs_iters=5000, **v)))
jobs.sort(key=lambda j: -j['community']['n_species'])
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp07.jsonl')
