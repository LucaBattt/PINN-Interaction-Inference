"""Builds Hidden-Species-Demo.ipynb (run it with: jupyter nbconvert --to notebook --execute --inplace Hidden-Species-Demo.ipynb)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# Inferring the interaction network when one species is barely measured (or not measured at all)

This notebook uses `hs_pinn.py`, which reuses the simulation helpers and the training curriculum of `PINN-Inference.ipynb`
(data-only warm-up → least-squares reset of $(r, A)$ → physics-weight ramp → refinement) with these changes:

1. **Masked data loss**: a species can have only a few data points, or none. The physics loss is still applied to *all* species, so the network has to produce a trajectory for the unmeasured species that is consistent with the gLV equations and with what the measured species do.
2. **Positive outputs and log-form residual**: $x = \\exp(\\text{net})$ and the residual is $\\frac{d \\ln x_i}{dt} - (r_i + \\sum_j A_{ij} x_j)$. Without this, the unmeasured species can collapse to $x = 0$, which satisfies the standard residual trivially.
3. **L-BFGS polish** (double precision) after Adam. This is what makes the partially observed problem work: Adam alone stops far from the solution.
4. Entries of $r$ and $A$ can be **frozen** (e.g. known growth rates).
5. The per-IC PINNs are evaluated as one batched network with analytic time derivatives (same architecture, faster on CPU)."""),
code("""import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import hs_pinn as hs
from showcase import plot_trajectories, plot_matrices

torch.set_num_threads(4)
np.set_printoptions(precision=3, suppress=True)"""),
md("""# Example 1: one species measured at only 3 time points

Five species, ten initial conditions, simulated exactly as in `PINN-Inference.ipynb`. The species with the strongest effect on the others is measured at 3 time points per trajectory (start, middle, end); the other four are measured at all 101 points. Gaussian noise with std 0.01 is added to every measurement."""),
code("""n_species, RSEED, noise_std, n_pts = 5, 18, 0.01, 3

com = hs.make_community(n_species, seed=RSEED)          # r, A, initial conditions, time grid and clean trajectories
sparse = hs.most_influential_species(com)               # index of the species that will be barely measured
print(f"Sparse species: sp{sparse + 1}")
print(f"Growth rates:\\n{com['r']}\\nInteraction matrix:\\n{com['A']}")"""),
code("""res = hs.train(com, hidden=sparse, n_hidden_pts=n_pts, noise_std=noise_std, lbfgs_iters=3000, log_every=1000)"""),
md("""The model never sees the gray curve below, only the three black points. The red curve is the PINN output for that species, which is determined by the physics loss."""),
code("""plot_trajectories(com, res, dpi=120); plt.show()"""),
code("""plot_matrices(com, res, dpi=120); plt.show()"""),
md("""Comparison with what gradient matching can do with the same data: (a) interpolate the three points and regress, (b) leave the species out of the model. Both baselines get smoothing-spline derivatives that use the true noise level. Errors are relative Frobenius errors in %."""),
code("""m = res['metrics']
b_int = hs.baseline_interpolate_hidden(com, sparse, n_pts, noise_std)
b_ign = hs.baseline_ignore_hidden(com, sparse, noise_std)
rows = {
    'PINN (3 points for the sparse species)': [m['A_err'], m['A_obs_block_err'], m['A_hidden_col_err'], m['A_hidden_row_err'], m['sign_acc']],
    'Gradient matching, 3 points interpolated': [b_int['A_err'], b_int['A_obs_block_err'], b_int['A_hidden_col_err'], b_int['A_hidden_row_err'], b_int['sign_acc']],
    'Gradient matching, species left out': [np.nan, b_ign['A_obs_block_err'], np.nan, np.nan, np.nan],
}
table = pd.DataFrame(rows, index=['A (all)', 'A between fully observed species', 'effect of sparse species on others',
                                  'effect of others on sparse species', 'sign accuracy']).T
table.iloc[:, :4] = (100 * table.iloc[:, :4]).round(1)
table"""),
md("""# Example 2: one species with no data at all, growth rates known

Four species; the most influential one is never measured. The growth rates $r$ are given (not trained). The abundance scale of a species that is never measured cannot be identified (multiplying $x_h$ by $c$ and dividing its column of $A$ by $c$ gives the same observations), so its self-regulation is frozen at $A_{hh} = -1$, which is the value used by the generator.

**How to tell whether a run can be trusted:** `sim_fit_rmse` integrates the inferred model from the inferred initial state and compares it with the available data. It only uses the data, so it also works on real experiments. A good run has `sim_fit_rmse` close to the noise level; a clearly larger value means the run ended in a wrong solution and should be restarted with another seed (`hs.train_with_restarts` does this automatically)."""),
code("""n_species, RSEED, noise_std = 4, 4, 0.01

com2 = hs.make_community(n_species, seed=RSEED)
hidden = hs.most_influential_species(com2)
res2 = hs.train(com2, hidden=hidden, n_hidden_pts=0, r_known=True, noise_std=noise_std, lbfgs_iters=3000, log_every=1000)
m2 = res2['metrics']
print(f"\\nsim_fit_rmse = {m2['sim_fit_rmse']:.4f}   (noise std = {noise_std})")
print(f"Relative error on A: {100 * m2['A_err']:.1f}%   |   on the hidden species' trajectory: {100 * m2['x_hidden_err']:.1f}%   |   sign accuracy: {m2['sign_acc']:.2f}")"""),
code("""plot_trajectories(com2, res2, dpi=120); plt.show()"""),
code("""plot_matrices(com2, res2, dpi=120); plt.show()"""),
]
nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
nbf.write(nb, 'Hidden-Species-Demo.ipynb')
print('written')
