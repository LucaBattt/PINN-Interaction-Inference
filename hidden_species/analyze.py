"""Aggregates the JSON-lines results into tables (printed as Markdown)."""
import sys, json
import numpy as np
import hs_pinn as hs
from runner import load


def fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:5.1f}"


def table_sparse(path, noise_std=0.0):
    rows = [r for r in load(path) if r['train'].get('noise_std', 0.0) == noise_std]
    print(f"\n| community | sparse sp. | pts | PINN: A all | obs block | sparse col | sparse row | x sparse | sign acc | "
          f"interp. baseline: A all | obs block | col | row | ignore baseline: obs block |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows.sort(key=lambda r: (r['community']['n_species'], r['community']['seed'], r['train'].get('n_hidden_pts', 0)))
    for r in rows:
        c, tr, m = r['community'], r['train'], r['metrics']
        com = hs.make_community(**c)
        h, pts = tr['hidden'], tr.get('n_hidden_pts', 0)
        b = hs.baseline_interpolate_hidden(com, h, pts, noise_std, tr.get('noise_seed', 42)) if pts > 0 else {}
        b0 = hs.baseline_ignore_hidden(com, h, noise_std, tr.get('noise_seed', 42))
        print(f"| n={c['n_species']} seed {c['seed']} | {h} | {pts} | {fmt(m['A_err'])} | {fmt(m['A_obs_block_err'])} | {fmt(m['A_hidden_col_err'])} | "
              f"{fmt(m['A_hidden_row_err'])} | {fmt(m['x_hidden_err'])} | {m['sign_acc']:.2f} | {fmt(b.get('A_err'))} | {fmt(b.get('A_obs_block_err'))} | "
              f"{fmt(b.get('A_hidden_col_err'))} | {fmt(b.get('A_hidden_row_err'))} | {fmt(b0['A_obs_block_err'])} |")


def summary(path, key='n_hidden_pts', noise_std=0.0, extra=lambda r: True):
    rows = [r for r in load(path) if r['train'].get('noise_std', 0.0) == noise_std and extra(r)]
    groups = {}
    for r in rows:
        groups.setdefault((r['community']['n_species'], r['train'].get(key, 0)), []).append(r['metrics'])
    print(f"\n| n species | {key} | runs | median A err % | max A err % | median obs block % | median sparse col % | median sparse row % | median x sparse % | mean sign acc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (n, k), ms in sorted(groups.items()):
        g = lambda name: np.array([m.get(name, np.nan) for m in ms])
        print(f"| {n} | {k} | {len(ms)} | {fmt(float(np.median(g('A_err'))))} | {fmt(float(np.max(g('A_err'))))} | {fmt(float(np.median(g('A_obs_block_err'))))} | "
              f"{fmt(float(np.median(g('A_hidden_col_err'))))} | {fmt(float(np.median(g('A_hidden_row_err'))))} | {fmt(float(np.median(g('x_hidden_err'))))} | {np.mean(g('sign_acc')):.2f} |")




def table_noise(path):
    """PINN with a sparse species vs PINN with everything observed vs gradient matching, per noise level."""
    rows = load(path)
    print("\n| n species | noise std | PINN, all observed | PINN, 10 pts | PINN, 3 pts | grad. matching, all observed | grad. matching, 10 pts interp. | grad. matching, 3 pts interp. |")
    print("|---|---|---|---|---|---|---|---|")
    groups = {}
    for r in rows:
        key = (r['community']['n_species'], r['train'].get('noise_std', 0.0))
        g = groups.setdefault(key, dict(full=[], p10=[], p3=[], b_full=[], b10=[], b3=[]))
        tr = r['train']
        com = hs.make_community(**r['community'])
        if tr.get('hidden') is None:
            g['full'].append(r['metrics']['A_err'])
            g['b_full'].append(hs.baseline_full(com, tr['noise_std'])['A_err'])
        else:
            g['p%d' % tr['n_hidden_pts']].append(r['metrics']['A_err'])
            g['b%d' % tr['n_hidden_pts']].append(hs.baseline_interpolate_hidden(com, tr['hidden'], tr['n_hidden_pts'], tr['noise_std'])['A_err'])
    med = lambda v: "  -  " if not v else f"{100 * np.median(v):.1f} ({100 * np.min(v):.1f}-{100 * np.max(v):.1f})"
    for (n, noise), g in sorted(groups.items()):
        print(f"| {n} | {noise} | {med(g['full'])} | {med(g['p10'])} | {med(g['p3'])} | {med(g['b_full'])} | {med(g['b10'])} | {med(g['b3'])} |")


if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == 'noise':
        table_noise(path)
    else:
        noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        table_sparse(path, noise)
        summary(path, noise_std=noise)
