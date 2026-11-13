"""Plots for one sparse-species run: reconstructed trajectory of the sparse species, and true vs inferred A."""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
import hs_pinn as hs
from scipy.interpolate import PchipInterpolator


def plot_trajectories(com, res, max_ic=5, dpi=150):
    n, h = com['n_species'], res['cfg']['hidden']
    t, X, Xn, xp, mask = com['t'], com['X'], res['X_noisy'], res['x_pred'], res['mask']
    idx = np.where(mask[:, h])[0]
    show = list(range(min(X.shape[0], max_ic)))
    others = [i for i in range(n) if i != h]
    fig, ax = plt.subplots(2, len(show), figsize=(3.4 * len(show), 5.6), dpi=dpi, sharex=True, squeeze=False)
    for c, k in enumerate(show):
        for i in others:
            ax[0, c].scatter(t[::3], Xn[k, ::3, i], s=7, alpha=0.45)
            ax[0, c].plot(t, xp[k, :, i], lw=1, color='black', ls='--')
        ax[0, c].set_title(f"IC {k + 1}", fontsize=10)
        ax[0, c].grid(True, ls='--', alpha=0.3)
        ax[1, c].plot(t, X[k, :, h], lw=4, color='#b9c7d8', label='true (not given to the model)')
        ax[1, c].plot(t, xp[k, :, h], lw=1.5, color='#c0392b', label='PINN reconstruction')
        if len(idx) > 1:
            f = PchipInterpolator(t[idx], Xn[k, idx, h])
            ax[1, c].plot(t, f(t), lw=1, color='gray', ls=':', label='interpolation of the points')
        if len(idx) > 0:
            ax[1, c].scatter(t[idx], Xn[k, idx, h], s=45, color='black', zorder=5, label=f'available data ({len(idx)} per trajectory)')
        ax[1, c].set_xlabel('time')
        ax[1, c].grid(True, ls='--', alpha=0.3)
    ax[0, 0].set_ylabel(f'{n - 1} fully observed species\n(dots: data, dashed: PINN)')
    ax[1, 0].set_ylabel(f"{'sparse' if len(idx) else 'hidden'} species (sp{h + 1})")
    ax[1, 0].legend(fontsize=7, loc='best')
    what = f"measured at {len(idx)} time point(s) per trajectory" if len(idx) else "never measured"
    fig.suptitle(f"{n} species; species {h + 1} {what}; noise std {res['cfg']['noise_std']}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_matrices(com, res, dpi=150):
    n, h, m = com['n_species'], res['cfg']['hidden'], res['metrics']
    mats = [('True A', com['A']), (f"PINN (relative error {100 * m['A_err']:.1f}%)", res['A'])]
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=dpi)
    vmax = np.abs(com['A']).max()
    labels = [f"sp{j + 1}" + ('*' if j == h else '') for j in range(n)]
    for a, (title, M) in zip(ax, mats):
        a.imshow(M, cmap='RdBu', vmin=-vmax, vmax=vmax)
        for i in range(n):
            for j in range(n):
                a.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=7)
        a.set_title(title, fontsize=10)
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels(labels, fontsize=8); a.set_yticklabels(labels, fontsize=8)
    off = ~np.eye(n, dtype=bool)
    involves_h = (np.arange(n)[:, None] == h) | (np.arange(n)[None, :] == h)
    ax[2].plot([-vmax, vmax], [-vmax, vmax], color='gray', lw=0.8)
    ax[2].scatter(com['A'][off & ~involves_h], res['A'][off & ~involves_h], c='#2c3e50', s=28, label='between fully observed species')
    ax[2].scatter(com['A'][off & involves_h], res['A'][off & involves_h], c='#c0392b', s=28, label='involving the sparse / hidden species (*)')
    ax[2].set_xlabel('true A_ij (off-diagonal)'); ax[2].set_ylabel('inferred A_ij')
    ax[2].legend(fontsize=7); ax[2].grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import torch, matplotlib
    matplotlib.use('Agg')
    n, seed, pts, noise, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 4)
    com = hs.make_community(n, seed=seed)
    h = hs.most_influential_species(com)
    res = hs.train(com, hidden=h, n_hidden_pts=pts, noise_std=noise, lbfgs_iters=5000 if noise == 0 else 3000)
    print({k: round(v, 4) for k, v in res['metrics'].items()})
    plot_trajectories(com, res, dpi=200).savefig(out + '_trajectories.png')
    plot_matrices(com, res, dpi=200).savefig(out + '_matrix.png')
    json.dump(dict(metrics=res['metrics'], A=res['A'].tolist(), r=res['r'].tolist(), hidden=h), open(out + '.json', 'w'))
