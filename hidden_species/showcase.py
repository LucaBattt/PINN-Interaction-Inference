"""Trains one sparse-species case and draws: (1) the reconstructed trajectory of the sparse species, (2) true vs inferred A."""
import sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hs_pinn as hs
from scipy.interpolate import PchipInterpolator


def main(n=5, seed=18, pts=3, noise=0.0, out='results/figs/showcase'):
    torch.set_num_threads(4)
    com = hs.make_community(n, seed=seed)
    h = hs.most_influential_species(com)
    res = hs.train(com, hidden=h, n_hidden_pts=pts, noise_std=noise, lbfgs_iters=5000 if noise == 0 else 3000)
    m = res['metrics']
    print({k: round(v, 4) for k, v in m.items()})
    t, X, Xn, xp, mask = com['t'], com['X'], res['X_noisy'], res['x_pred'], res['mask']
    idx = np.where(mask[:, h])[0]
    K = X.shape[0]
    show = list(range(min(K, 5)))

    # ---- Figure 1: trajectories
    fig, ax = plt.subplots(2, len(show), figsize=(3.4 * len(show), 5.6), dpi=200, sharex=True)
    others = [i for i in range(n) if i != h]
    for c, k in enumerate(show):
        for i in others:
            ax[0, c].scatter(t[::3], Xn[k, ::3, i], s=7, alpha=0.45)
            ax[0, c].plot(t, xp[k, :, i], lw=1, color='black', ls='--')
        ax[0, c].set_title(f"IC {k + 1}", fontsize=10)
        ax[0, c].grid(True, ls='--', alpha=0.3)
        ax[1, c].plot(t, X[k, :, h], lw=4, color='#b9c7d8', label='true (never shown to the model)')
        ax[1, c].plot(t, xp[k, :, h], lw=1.5, color='#c0392b', label='PINN reconstruction')
        if len(idx) > 1:
            f = PchipInterpolator(t[idx], Xn[k, idx, h])
            ax[1, c].plot(t, f(t), lw=1, color='gray', ls=':', label='interpolation of the points')
        ax[1, c].scatter(t[idx], Xn[k, idx, h], s=45, color='black', zorder=5, label=f'available data ({len(idx)} points)')
        ax[1, c].set_xlabel('time')
        ax[1, c].grid(True, ls='--', alpha=0.3)
    ax[0, 0].set_ylabel(f'fully observed species ({n - 1})\ndots: data, dashed: PINN')
    ax[1, 0].set_ylabel(f'sparse species (sp{h + 1})')
    ax[1, 0].legend(fontsize=7, loc='best')
    fig.suptitle(f"{n} species, species {h + 1} measured at {len(idx)} time point(s) per trajectory, noise std {noise}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out + '_trajectories.png')

    # ---- Figure 2: interaction matrices
    base = hs.baseline_interpolate_hidden(com, h, pts, noise) if pts > 0 else None
    mats = [('True A', com['A']), (f"PINN (error {100 * m['A_err']:.1f}%)", res['A'])]
    fig, ax = plt.subplots(1, len(mats) + 1, figsize=(4.2 * (len(mats) + 1), 3.8), dpi=200)
    vmax = np.abs(com['A']).max()
    for a, (title, M) in zip(ax, mats):
        im = a.imshow(M, cmap='RdBu', vmin=-vmax, vmax=vmax)
        for i in range(n):
            for j in range(n):
                a.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center', fontsize=7)
        a.set_title(title, fontsize=10)
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels([f"sp{j + 1}" + ('*' if j == h else '') for j in range(n)], fontsize=8)
        a.set_yticklabels([f"sp{j + 1}" + ('*' if j == h else '') for j in range(n)], fontsize=8)
    off = ~np.eye(n, dtype=bool)
    ax[-1].plot([-vmax, vmax], [-vmax, vmax], color='gray', lw=0.8)
    colors = np.where((np.arange(n)[:, None] == h) | (np.arange(n)[None, :] == h), '#c0392b', '#2c3e50')
    ax[-1].scatter(com['A'][off], res['A'][off], c=colors[off], s=28)
    ax[-1].set_xlabel('true A_ij (off-diagonal)'); ax[-1].set_ylabel('inferred A_ij')
    ax[-1].set_title('red: interactions involving the sparse species (*)', fontsize=9)
    ax[-1].grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + '_matrix.png')
    json.dump(dict(metrics=m, A=res['A'].tolist(), r=res['r'].tolist(), hidden=h), open(out + '.json', 'w'))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]), float(a[3]), a[4])
