"""Prints the summary numbers quoted in README.md from the result files."""
import warnings
import numpy as np
import hs_pinn as hs
from runner import load
warnings.filterwarnings('ignore')


def s(v):
    v = np.array(v) * 100
    return f"{np.median(v):.1f} ({v.min():.1f}-{v.max():.1f})"


def com_of(r):
    return hs.make_community(**r['community'])


def key(r):
    return (r['community']['n_species'], r['community']['seed'])


def best_by_fit(runs):
    """One run per community: the one whose simulated model fits the available data best."""
    best = {}
    for r in runs:
        if key(r) not in best or r['metrics']['sim_fit_rmse'] < best[key(r)]['metrics']['sim_fit_rmse']:
            best[key(r)] = r
    return [best[k] for k in sorted(best)]


print("=== A. Sparse species = most influential, no noise (exp06). Relative error on A in %: median (min-max) over 9 communities")
rs = load('results/exp06.jsonl')
for m in [1, 3, 5, 10]:
    g = [r for r in rs if r['train']['n_hidden_pts'] == m]
    b = [hs.baseline_interpolate_hidden(com_of(r), r['train']['hidden'], m)['A_err'] for r in g]
    print(f"  {m:2d} pts | PINN {s([r['metrics']['A_err'] for r in g])} | interpolate+regress {s(b)}")
print("  species left out, error on the block between observed species:", s([hs.baseline_ignore_hidden(com_of(r), r['train']['hidden'])['A_obs_block_err'] for r in rs if r['train']['n_hidden_pts'] == 1]))

print("=== B. Sparse species = least influential (exp09)")
rs = load('results/exp09.jsonl')
for m, noise in [(1, 0), (3, 0), (10, 0), (10, 0.01)]:
    g = [r for r in rs if r['train']['n_hidden_pts'] == m and r['train'].get('noise_std', 0) == noise]
    b = [hs.baseline_interpolate_hidden(com_of(r), r['train']['hidden'], m, noise)['A_err'] for r in g]
    v = np.array([r['metrics']['A_err'] for r in g])
    print(f"  {m:2d} pts noise {noise} | PINN A {s(v)}, under 5%: {(v < 0.05).sum()}/9 | block between observed species {s([r['metrics']['A_obs_block_err'] for r in g])} | interpolate+regress {s(b)}")

print("=== C. Noise, sparse species = most influential (exp08). Per noise level over 9 communities")
rs = load('results/exp08.jsonl')
for noise in [0.01, 0.03]:
    full = [r for r in rs if r['train'].get('hidden') is None and r['train']['noise_std'] == noise]
    print(f"  noise {noise} | PINN all observed {s([r['metrics']['A_err'] for r in full])} | regression all observed {s([hs.baseline_full(com_of(r), noise)['A_err'] for r in full])}")
    for m in [10, 3]:
        g = [r for r in rs if r['train'].get('hidden') is not None and r['train']['n_hidden_pts'] == m and r['train']['noise_std'] == noise]
        b = [hs.baseline_interpolate_hidden(com_of(r), r['train']['hidden'], m, noise)['A_err'] for r in g]
        print(f"     {m:2d} pts | PINN A {s([r['metrics']['A_err'] for r in g])} | block between observed {s([r['metrics']['A_obs_block_err'] for r in g])} | sparse trajectory {s([r['metrics']['x_hidden_err'] for r in g])} "
              f"| sign acc {np.mean([r['metrics']['sign_acc'] for r in g]):.2f} | interpolate+regress {s(b)}")

print("=== D. Fully hidden species (exp07, exp10, exp11)")
rs = load('results/exp07.jsonl') + load('results/exp10.jsonl') + load('results/exp11.jsonl')
for variant in ['nothing known', 'r known', 'r + self-regulation known']:
    for noise in [0.0, 0.01, 0.03]:
        g = [r for r in rs if r['variant'] == variant and r['train'].get('noise_std', 0.0) == noise]
        if not g:
            continue
        first = [r for r in g if r['train'].get('seed', 42) == 42]
        best = best_by_fit(g)
        for name, sel in [('single run', first), ('best of up to 3 restarts (by fit to data)', best)]:
            v = np.array([r['metrics']['A_err'] for r in sel])
            print(f"  {variant:<26s} noise {noise} | {name:<42s} | A {s(v)}, under 10%: {(v < 0.10).sum()}/{len(v)} | hidden trajectory {s([r['metrics']['x_hidden_err'] for r in sel])} "
                  f"| sign acc {np.mean([r['metrics']['sign_acc'] for r in sel]):.2f}")
        print("      per community (best):", [f"{100 * r['metrics']['A_err']:.1f}" for r in best], "restarts run:", len(g) - len(first))
