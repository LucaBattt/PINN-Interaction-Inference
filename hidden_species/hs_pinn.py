"""
Hidden-species experiments for the PINN gLV inference pipeline.

The simulation helpers (rk4, integrador_ode, generate_glv, glv, physics_loss_weight) are copied from
PINN-Inference.ipynb. The training loop follows the same curriculum as the notebook
(data-only warm-up -> least-squares reset of (r, A) -> physics-weight ramp -> refinement with cosine LR decay),
with three changes that make partially observed communities possible:

  1. The data loss is masked: a species can have no data at all, or only a few time points.
  2. Entries of r and A can be frozen (known growth rates, gauge-fixed self-regulation of the hidden species).
  3. The per-IC PINNs are evaluated as one batched network with analytic time derivatives (same architecture
     as the notebook's PINN class, only faster on CPU).
"""
import time
import numpy as np
import torch
import torch.nn as nn



# ----------------------------------------------------------------------------------------------------
# Copied from PINN-Inference.ipynb
# ----------------------------------------------------------------------------------------------------
def rk4(f, x, t, h, p):
    k1 = f(x, t, p)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h, p)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h, p)
    k4 = f(x + h * k3, t + h, p)
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrador_ode(m, f, xa, t_0, t_f, t_total_steps, p, c=lambda x, t, p: x):
    k = t_total_steps
    a, b = t_0, t_f
    assert k > 0
    n = len(xa)
    h = (b - a) / k
    w = np.zeros((n, k + 1))
    t = np.zeros(k + 1)
    w[:, 0] = xa
    t[0] = a
    max_magnitude, min_magnitude = 1e20, -1e20
    for j in range(k):
        t[j + 1] = t[j] + h
        try:
            next_w = m(f, w[:, j], t[j], h, p)
            next_w = c(next_w, t[j + 1], p)
            if (not np.all(np.isfinite(next_w))) or np.any(next_w > max_magnitude) or np.any(next_w < min_magnitude):
                raise RuntimeError("Divergence detected")
            w[:, j + 1] = next_w
        except (RuntimeError, FloatingPointError):
            w[:, j + 1] = w[:, j]
    return t, w.T


def generate_glv(n, r_mean=0.0, r_std=1.0, A_mean=0.0, A_std=1.0, sparsity=None, self_reg_strength=None, seed=None, uniform=False):
    rng = np.random.RandomState(seed)
    if uniform:
        r = rng.uniform(-4, 4, size=(n,))
        A = rng.uniform(-4, 4, size=(n, n))
    else:
        r = rng.normal(loc=r_mean, scale=r_std, size=(n,))
        A = rng.normal(loc=A_mean, scale=A_std, size=(n, n))
    if sparsity is not None and 0.0 < sparsity < 1.0:
        mask = rng.rand(n, n) < sparsity
        np.fill_diagonal(mask, False)
        A[mask] = 0.0
    if self_reg_strength is not None:
        np.fill_diagonal(A, -np.abs(self_reg_strength))
    return r, A


def glv(x, t, p):
    r, A = p[0], p[1]
    return x * (r + A.dot(x))


def physics_loss_weight(w_data, w_data_dot, t):
    std_w = np.std(w_data, axis=-2)
    std_dw_dt = np.std(w_data_dot, axis=-2)
    return np.mean((std_w / std_dw_dt) ** 2)


# ----------------------------------------------------------------------------------------------------
# Community simulation
# ----------------------------------------------------------------------------------------------------
def make_community(n_species, seed, num_init_cond=None, T=15.0, t_step=0.15, sparsity=0.3,
                   r_mean=1.0, r_std=0.8, A_std=0.5, ic_seed=None):
    """Simulates a gLV community exactly as PINN-Inference.ipynb does (same generator and RK4 integrator)."""
    if num_init_cond is None:
        num_init_cond = 2 * n_species
    r, A = generate_glv(n=n_species, r_mean=r_mean, r_std=r_std, A_mean=0.0, A_std=A_std,
                        sparsity=sparsity, self_reg_strength=1.0, seed=seed)
    rng = np.random.RandomState(seed if ic_seed is None else ic_seed)
    x0 = [rng.uniform(low=0.1, high=1.5, size=(n_species,)) for _ in range(num_init_cond)]
    steps = int(round(T / t_step))
    X = []
    for k in range(num_init_cond):
        t, w = integrador_ode(rk4, glv, xa=x0[k], t_0=0.0, t_f=T, t_total_steps=steps, p=[r, A])
        X.append(w)
    return dict(r=r, A=A, x0=np.array(x0), t=t, X=np.array(X), n_species=n_species, seed=seed)


# ----------------------------------------------------------------------------------------------------
# Batched PINN: num_init_cond independent copies of the notebook's PINN (tanh MLP), with analytic d/dt
# ----------------------------------------------------------------------------------------------------
class BatchedPINN(nn.Module):
    def __init__(self, n_nets, n_output, n_hidden=3, n_hidden_neurons=32):
        super().__init__()
        sizes = [1] + [n_hidden_neurons] * n_hidden + [n_output]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = 1.0 / np.sqrt(fan_in)  # same as nn.Linear's default initialisation
            self.W.append(nn.Parameter(torch.empty(n_nets, fan_in, fan_out).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(n_nets, 1, fan_out).uniform_(-bound, bound)))

    def forward(self, t):
        """t: (n_points, 1). Returns the raw outputs u and du/dt, both of shape (n_nets, n_points, n_output)."""
        h = t.unsqueeze(0) @ self.W[0] + self.b[0]
        a = torch.tanh(h)
        da = (1 - a * a) * self.W[0]
        for W, b in zip(list(self.W)[1:-1], list(self.b)[1:-1]):
            h = a @ W + b
            dh = da @ W
            a = torch.tanh(h)
            da = (1 - a * a) * dh
        return a @ self.W[-1] + self.b[-1], da @ self.W[-1]


def net_outputs(net, t, positive):
    """Abundances x, dx/dt and d(ln x)/dt predicted by the PINNs."""
    u, du = net(t)
    if positive:            # x = exp(u): abundances are positive by construction
        x = torch.exp(u)
        return x, x * du, du
    return u, du, du / u    # notebook parametrisation (x = u)


# ----------------------------------------------------------------------------------------------------
# Least-squares (r, A) for fixed PINNs, honouring frozen entries
# ----------------------------------------------------------------------------------------------------
def lstsq_r_A(x, dx, dlogx, r0, A0, r_free, A_free, resid):
    """
    Generalises the notebook's lstsq_r_A: for each species, regresses the PINN derivative on the gLV features.
    Frozen entries (r_free / A_free False) keep the values in r0 / A0 and are moved to the left-hand side.
    x, dx, dlogx: np.ndarray (n_points_total, n_species)
    """
    n = x.shape[1]
    r, A = r0.copy(), A0.copy()
    for i in range(n):
        if resid == 'log':     # d ln x_i/dt = r_i + sum_j A_ij x_j
            feats = np.column_stack([np.ones(len(x))] + [x[:, j] for j in range(n)])
            y = dlogx[:, i]
        else:                  # dx_i/dt = r_i x_i + sum_j A_ij x_i x_j
            feats = np.column_stack([x[:, i]] + [x[:, i] * x[:, j] for j in range(n)])
            y = dx[:, i]
        coef0 = np.concatenate([[r0[i]], A0[i]])
        free = np.concatenate([[r_free[i]], A_free[i]])
        y = y - feats[:, ~free] @ coef0[~free]
        if free.any():
            sol = np.linalg.lstsq(feats[:, free], y, rcond=None)[0]
            coef0[free] = sol
        r[i], A[i] = coef0[0], coef0[1:]
    return r, A


# ----------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------
DEFAULTS = dict(
    # observation model
    hidden=None,              # index of the partially observed species (None: all species observed)
    n_hidden_pts=0,           # number of time points available for that species (0: no data at all, 1: only t = 0)
    noise_std=0.0,            # std of the additive Gaussian noise
    noise_seed=42,
    # prior knowledge
    r_known=False,            # growth rates given (not trained)
    fix_hidden_selfreg=True,  # with a fully hidden species: freeze A_hh = -1 (fixes the abundance scale)
    diag_known=False,         # all self-regulation terms A_ii given
    # PINN
    positive=True,            # x = exp(net) instead of x = net
    resid='log',              # 'log': d ln x/dt - (r + A x) ; 'standard': dx/dt - x (r + A x)  (notebook)
    n_hidden_layers=3, n_hidden_neurons=32, n_coloc=400,
    # curriculum (same structure as the notebook)
    warmup_epochs=2000, ramp_epochs=3000, refine_epochs=1000,
    alpha_start=1e-4, alpha_end=1e-1,
    lr_net=1e-3, lr_net_min=1e-5, lr_xi=1e-3, lr_xi_min=1e-5,
    beta=0.0,                 # L1 penalty on the off-diagonal entries of A
    varpro=False,             # variable projection: (r, A) are not trained by Adam; at every epoch they are the
                              # least-squares solution for the current PINNs (the physics loss is differentiated through it)
    lbfgs_iters=2000,         # L-BFGS polish (float64) after Adam, on the same loss as the refinement phase
    oracle_warmup=False,      # DIAGNOSTIC ONLY: hidden species' true trajectory is used during the data-only phase, then removed
    seed=42, log_every=0,
)


def observation_mask(n_t, n_species, hidden, n_hidden_pts):
    """Boolean (n_t, n_species): which data points enter the data loss."""
    mask = np.ones((n_t, n_species), dtype=bool)
    if hidden is not None:
        mask[:, hidden] = False
        if n_hidden_pts > 0:
            idx = np.unique(np.round(np.linspace(0, n_t - 1, n_hidden_pts)).astype(int)) if n_hidden_pts > 1 else np.array([0])
            mask[idx, hidden] = True
    return mask


def train(com, **kwargs):
    cfg = {**DEFAULTS, **kwargs}
    unknown = set(kwargs) - set(DEFAULTS)
    assert not unknown, f"Unknown options: {unknown}"
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_default_dtype(torch.float32)

    n, t, X_clean = com['n_species'], com['t'], com['X']
    K, n_t = X_clean.shape[0], X_clean.shape[1]
    h = cfg['hidden']
    fully_hidden = h is not None and cfg['n_hidden_pts'] == 0

    # Data (+ noise) and observation mask
    rng = np.random.RandomState(cfg['noise_seed'])
    X_noisy = X_clean + rng.normal(0, 1, X_clean.shape) * cfg['noise_std']
    mask_np = observation_mask(n_t, n, h, cfg['n_hidden_pts'])
    obs_species = [i for i in range(n) if not (h == i)]

    # Physics loss weight (notebook formula, on the fully observed species)
    X_obs = X_noisy[:, :, obs_species]
    X_obs_dot = np.gradient(X_obs, t, axis=1)
    if cfg['resid'] == 'log':
        alpha = np.mean((np.std(X_obs, axis=1) / np.std(X_obs_dot / np.clip(X_obs, 0.05, None), axis=1)) ** 2)
    else:
        alpha = physics_loss_weight(X_obs, X_obs_dot, t)

    # Trainable / frozen entries of (r, A)
    r_free = np.ones(n, dtype=bool)
    A_free = np.ones((n, n), dtype=bool)
    r_fixed, A_fixed = np.zeros(n), np.zeros((n, n))
    if cfg['r_known']:
        r_free[:] = False
        r_fixed = com['r'].copy()
    if cfg['diag_known']:
        A_free[np.diag_indices(n)] = False
        A_fixed[np.diag_indices(n)] = np.diag(com['A'])
    if fully_hidden and cfg['fix_hidden_selfreg']:
        A_free[h, h] = False
        A_fixed[h, h] = -1.0

    r_param = torch.tensor(np.where(r_free, np.random.uniform(-1, 1, n), r_fixed), dtype=torch.float32, requires_grad=True)
    A_param = torch.tensor(np.where(A_free, np.random.uniform(-1, 1, (n, n)), A_fixed), dtype=torch.float32, requires_grad=True)
    net = BatchedPINN(K, n, cfg['n_hidden_layers'], cfg['n_hidden_neurons'])

    # All constant tensors live in T, so that the whole problem can be moved to float64 for the L-BFGS polish
    T = {}

    def build_tensors(dt):
        mask = torch.tensor(mask_np).unsqueeze(0).expand(K, -1, -1)
        t_data = torch.tensor(t, dtype=dt).reshape(-1, 1)
        t_coloc = torch.linspace(float(t.min()), float(t.max()), cfg['n_coloc'], dtype=dt).reshape(-1, 1)
        r_free_t, A_free_t = torch.tensor(r_free), torch.tensor(A_free)
        r_fixed_t, A_fixed_t = torch.tensor(r_fixed, dtype=dt), torch.tensor(A_fixed, dtype=dt)
        T.update(dt=dt, mask=mask, t_data=t_data, t_coloc=t_coloc, t_all=torch.cat([t_data, t_coloc]),
                 X_data=torch.tensor(X_noisy, dtype=dt), X_clean=torch.tensor(X_clean, dtype=dt),
                 n_obs=mask.sum(dim=1).clamp(min=1).to(dt), has_data=(mask.sum(dim=1) > 0).to(dt), m0=mask[:, 0, :].to(dt),
                 r_free=r_free_t, A_free=A_free_t, r_fixed=r_fixed_t, A_fixed=A_fixed_t,
                 coef_free=torch.cat([r_free_t.reshape(-1, 1), A_free_t], dim=1),      # (n, n + 1): [r_i, A_i1 ... A_in]
                 coef_fixed=torch.cat([r_fixed_t.reshape(-1, 1), A_fixed_t], dim=1),
                 offdiag=~torch.eye(n, dtype=torch.bool))

    build_tensors(torch.float32)

    def current_r_A():
        return torch.where(T['r_free'], r_param, T['r_fixed']), torch.where(T['A_free'], A_param, T['A_fixed'])

    def projected_r_A(xc, dxc, dlogxc):
        """Least-squares (r, A) for the current PINN outputs (differentiable; frozen entries are respected)."""
        Xf = xc.reshape(-1, n)
        base = torch.cat([torch.ones(len(Xf), 1, dtype=T['dt']), Xf], dim=1)
        rows = []
        for i in range(n):
            if cfg['resid'] == 'log':
                F, y = base, dlogxc[..., i].reshape(-1)
            else:
                F, y = Xf[:, i:i + 1] * base, dxc[..., i].reshape(-1)
            free = T['coef_free'][i]
            y = y - F[:, ~free] @ T['coef_fixed'][i][~free]
            Ff = F[:, free].double()
            G = Ff.T @ Ff
            G = G + 1e-10 * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=torch.float64)
            sol = torch.linalg.solve(G, Ff.T @ y.double()).to(T['dt'])
            rows.append(T['coef_fixed'][i].masked_scatter(free, sol))
        C = torch.stack(rows)
        return C[:, 0], C[:, 1:]

    def losses(w_phys, oracle=False):
        x, dx, dlogx = net_outputs(net, T['t_all'], cfg['positive'])
        x_d = x[:, :n_t]
        if oracle:
            return ((x_d - T['X_clean']) ** 2).mean(dim=(1, 2)).sum(), None, None
        # Data loss: per-species MSE over the available points, summed over ICs
        # (a species with 3 data points weighs as much as one with 101)
        sq = ((x_d - T['X_data']) ** 2) * T['mask']
        loss_data = ((sq.sum(dim=1) / T['n_obs']) * T['has_data']).sum() / n
        # IC loss (as in the notebook), on the species observed at t = 0
        loss_ic = (((x_d[:, 0, :] - T['X_data'][:, 0, :]) ** 2) * T['m0']).sum() / n
        # Physics loss on the collocation points, for all species (also the hidden one)
        xc, dxc, dlogxc = x[:, n_t:], dx[:, n_t:], dlogx[:, n_t:]
        if cfg['varpro'] and w_phys > 0:
            r_c, A_c = projected_r_A(xc, dxc, dlogxc)
            with torch.no_grad():
                r_param.copy_(r_c)
                A_param.copy_(A_c)
        else:
            r_c, A_c = current_r_A()
        rhs = r_c + xc @ A_c.T
        res = (dlogxc - rhs) if cfg['resid'] == 'log' else (dxc - xc * rhs)
        loss_phys = (res ** 2).mean(dim=(1, 2)).sum()
        loss = loss_data + loss_ic + w_phys * loss_phys
        if cfg['beta'] > 0:
            loss = loss + cfg['beta'] * A_c[T['offdiag']].abs().sum()
        return loss, loss_data, loss_phys

    def A_err_now():
        return np.linalg.norm(current_r_A()[1].detach().double().numpy() - com['A']) / np.linalg.norm(com['A'])

    warm, ramp, refine = cfg['warmup_epochs'], cfg['ramp_epochs'], cfg['refine_epochs']
    total = warm + ramp + refine

    def progress(epoch, length):
        return min(1.0, max(0.0, (epoch - warm) / length))

    def alpha_schedule(epoch):
        if epoch < warm:
            return 0.0
        return cfg['alpha_start'] * (cfg['alpha_end'] / cfg['alpha_start']) ** progress(epoch, ramp)

    def lr_factor(epoch, lr0, lr_min):
        p = progress(epoch, ramp + refine)
        return (lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * p))) / lr0

    opt_net = torch.optim.Adam(net.parameters(), lr=cfg['lr_net'])
    opt_xi = torch.optim.Adam([r_param, A_param], lr=cfg['lr_xi'])

    history = dict(data=[], phys=[])
    t0 = time.time()
    for epoch in range(total):
        if epoch == warm:
            # End of the data-only phase: (r, A) set to their least-squares values for the current PINNs
            with torch.no_grad():
                x, dx, dlogx = net_outputs(net, T['t_coloc'], cfg['positive'])
                r_ls, A_ls = lstsq_r_A(x.reshape(-1, n).double().numpy(), dx.reshape(-1, n).double().numpy(),
                                       dlogx.reshape(-1, n).double().numpy(), r_fixed, A_fixed, r_free, A_free, cfg['resid'])
                r_param.copy_(torch.tensor(r_ls, dtype=torch.float32))
                A_param.copy_(torch.tensor(A_ls, dtype=torch.float32))
            opt_xi.state.clear()
        w_phys = alpha * alpha_schedule(epoch)
        for g in opt_net.param_groups:
            g['lr'] = cfg['lr_net'] * lr_factor(epoch, cfg['lr_net'], cfg['lr_net_min'])
        for g in opt_xi.param_groups:
            g['lr'] = cfg['lr_xi'] * lr_factor(epoch, cfg['lr_xi'], cfg['lr_xi_min'])
        opt_net.zero_grad()
        opt_xi.zero_grad()
        if cfg['oracle_warmup'] and epoch < warm:
            loss, _, _ = losses(0.0, oracle=True)
            loss.backward()
            opt_net.step()
            continue
        loss, loss_data, loss_phys = losses(w_phys)
        loss.backward()
        if not cfg['varpro']:
            opt_xi.step()
        opt_net.step()
        history['data'].append(loss_data.item())
        history['phys'].append(loss_phys.item())
        if cfg['log_every'] and (epoch % cfg['log_every'] == 0 or epoch == total - 1):
            print(f"  ep {epoch:6d}  data {loss_data.item():.2e}  phys {loss_phys.item():.2e}  w {w_phys:.2e}  A_err {A_err_now():.3f}", flush=True)
    seconds_adam = time.time() - t0

    if cfg['lbfgs_iters'] > 0:
        # L-BFGS polish in double precision, same loss as the refinement phase
        net.double()
        r_param.data, A_param.data = r_param.data.double(), A_param.data.double()
        build_tensors(torch.float64)
        w_phys = alpha * cfg['alpha_end']
        params = list(net.parameters()) + ([] if cfg['varpro'] else [r_param, A_param])
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg['lbfgs_iters'], max_eval=int(1.5 * cfg['lbfgs_iters']), history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-16, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss, _, _ = losses(w_phys)
            loss.backward()
            return loss
        opt.step(closure)
        if cfg['log_every']:
            print(f"  after L-BFGS: A_err {A_err_now():.3f}", flush=True)

    with torch.no_grad():
        _, loss_data, loss_phys = losses(alpha * cfg['alpha_end'])
        x_pred = net_outputs(net, T['t_data'], cfg['positive'])[0].double().numpy()
        r_c, A_c = current_r_A()
    torch.set_default_dtype(torch.float32)
    result = dict(r=r_c.double().numpy().copy(), A=A_c.double().numpy().copy(), x_pred=x_pred, X_noisy=X_noisy, mask=mask_np,
                  loss_data=loss_data.item(), loss_phys=loss_phys.item(), alpha=float(alpha),
                  history=history, cfg=cfg, seconds=time.time() - t0, seconds_adam=seconds_adam)
    result['metrics'] = evaluate(com, result)
    return result


# ----------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------
def rel_err(est, true):
    den = np.linalg.norm(true)
    return float(np.linalg.norm(est - true) / den) if den > 0 else float(np.linalg.norm(est - true))


def evaluate(com, result, zero_tol=1e-12):
    n, h = com['n_species'], result['cfg']['hidden']
    r_gt, A_gt = com['r'], com['A']
    r, A = result['r'].copy(), result['A'].copy()
    x_pred = result['x_pred'].copy()
    off = ~np.eye(n, dtype=bool)
    m = dict(loss_data=result['loss_data'], loss_phys=result['loss_phys'], seconds=result['seconds'], seconds_adam=result['seconds_adam'])

    # Does the inferred model, integrated from the inferred initial state, reproduce the observed species?
    t = com['t']
    sim = np.array([integrador_ode(rk4, glv, xa=x_pred[k, 0], t_0=t[0], t_f=t[-1], t_total_steps=len(t) - 1, p=[r, A])[1]
                    for k in range(x_pred.shape[0])])
    o_all = [i for i in range(n) if i != h]
    m['sim_obs_err'] = rel_err(sim[:, :, o_all], com['X'][:, :, o_all])
    # Same check against the data actually available (noisy, masked): usable to choose between restarts
    msk = np.broadcast_to(result['mask'], sim.shape)
    m['sim_fit_rmse'] = float(np.sqrt(np.mean((sim[msk] - result['X_noisy'][msk]) ** 2))) if np.all(np.isfinite(sim)) else float('inf')

    if h is not None and result['cfg']['n_hidden_pts'] == 0 and not result['cfg']['fix_hidden_selfreg']:
        # Abundance scale of a fully hidden species is not identifiable: express it in units where A_hh = -1
        s = -A[h, h]
        m['gauge_scale'] = float(s)
        if s > 0:
            x_pred[:, :, h] *= s
            A[:, h] /= s

    m['A_err'] = rel_err(A, A_gt)
    m['A_offdiag_err'] = rel_err(A[off], A_gt[off])
    m['r_err'] = rel_err(r, r_gt)
    nz = off & (np.abs(A_gt) > zero_tol)
    m['sign_acc'] = float(np.mean(np.sign(A[nz]) == np.sign(A_gt[nz])))
    if h is not None:
        o = [i for i in range(n) if i != h]
        m['A_obs_block_err'] = rel_err(A[np.ix_(o, o)], A_gt[np.ix_(o, o)])
        m['A_hidden_col_err'] = rel_err(A[o, h], A_gt[o, h])      # effect of the hidden species on the others
        m['A_hidden_row_err'] = rel_err(A[h, o], A_gt[h, o])      # effect of the others on the hidden species
        m['r_obs_err'] = rel_err(r[o], r_gt[o])
        m['r_hidden_err'] = rel_err(r[h], r_gt[h])
        m['x_hidden_err'] = rel_err(x_pred[:, :, h], com['X'][:, :, h])
        m['x_obs_err'] = rel_err(x_pred[:, :, o], com['X'][:, :, o])
    else:
        m['x_err'] = rel_err(x_pred, com['X'])
    return m


# ----------------------------------------------------------------------------------------------------
# Baselines: what a regression-based method can do without the hidden species' trajectory
# ----------------------------------------------------------------------------------------------------
def exact_log_derivative(com):
    return com['r'] + com['X'] @ com['A'].T


def baseline_ignore_hidden(com, hidden, noise_std=0.0, noise_seed=42):
    """
    Linear regression of d ln x_i/dt on [1, x_j (observed)] with the hidden species left out of the model.
    Noise-free: uses the exact derivatives (best case for any gradient-matching method).
    Noisy: derivatives from finite differences of the noisy data.
    Returns the relative error on the observed block of A, on r of the observed species, and the sign accuracy.
    """
    n = com['n_species']
    o = [i for i in range(n) if i != hidden]
    X = com['X']
    if noise_std > 0:
        rng = np.random.RandomState(noise_seed)
        X = X + rng.normal(0, 1, X.shape) * noise_std
        L = np.gradient(X, com['t'], axis=1) / np.clip(X, 0.02, None)
    else:
        L = exact_log_derivative(com)
    Xo = X[:, :, o].reshape(-1, n - 1)
    Lo = L[:, :, o].reshape(-1, n - 1)
    F = np.column_stack([np.ones(len(Xo)), Xo])
    coef = np.linalg.lstsq(F, Lo, rcond=None)[0]          # (n, n-1)
    r_hat, A_hat = coef[0], coef[1:].T
    A_gt = com['A'][np.ix_(o, o)]
    off = ~np.eye(n - 1, dtype=bool)
    nz = off & (np.abs(A_gt) > 1e-12)
    return dict(A_obs_block_err=rel_err(A_hat, A_gt), r_obs_err=rel_err(r_hat, com['r'][o]),
                sign_acc_obs=float(np.mean(np.sign(A_hat[nz]) == np.sign(A_gt[nz]))))


def baseline_interpolate_hidden(com, hidden, n_hidden_pts, noise_std=0.0, noise_seed=42):
    """
    Gradient matching with the sparse species filled in by interpolation (PCHIP through its few points; a constant if
    only one point is available). d ln x_i/dt is regressed on [1, x_1 ... x_n] for every species.
    Noise-free: exact derivatives for the fully observed species (best case); the sparse species' derivative comes from
    its interpolant. Noisy: finite differences of the noisy data.
    Returns the same metrics as evaluate().
    """
    from scipy.interpolate import PchipInterpolator
    n, t = com['n_species'], com['t']
    X = com['X'].copy()
    if noise_std > 0:
        rng = np.random.RandomState(noise_seed)
        X = X + rng.normal(0, 1, X.shape) * noise_std
        L = np.gradient(X, t, axis=1) / np.clip(X, 0.02, None)
    else:
        L = exact_log_derivative(com)
    idx = np.where(observation_mask(len(t), n, hidden, n_hidden_pts)[:, hidden])[0]
    for k in range(X.shape[0]):
        if len(idx) == 1:
            X[k, :, hidden] = X[k, idx[0], hidden]
            L[k, :, hidden] = 0.0
        else:
            f = PchipInterpolator(t[idx], np.clip(X[k, idx, hidden], 1e-3, None))
            X[k, :, hidden] = f(t)
            L[k, :, hidden] = f.derivative()(t) / np.clip(X[k, :, hidden], 1e-3, None)
    F = np.column_stack([np.ones(X.shape[0] * X.shape[1]), X.reshape(-1, n)])
    coef = np.linalg.lstsq(F, L.reshape(-1, n), rcond=None)[0]
    r_hat, A_hat = coef[0], coef[1:].T
    return _param_metrics(com, r_hat, A_hat, hidden)


def baseline_full(com, noise_std=0.0, noise_seed=42):
    """Gradient matching with every species fully observed (reference)."""
    n, t = com['n_species'], com['t']
    X = com['X'].copy()
    if noise_std > 0:
        rng = np.random.RandomState(noise_seed)
        X = X + rng.normal(0, 1, X.shape) * noise_std
        L = np.gradient(X, t, axis=1) / np.clip(X, 0.02, None)
    else:
        L = exact_log_derivative(com)
    F = np.column_stack([np.ones(X.shape[0] * X.shape[1]), X.reshape(-1, n)])
    coef = np.linalg.lstsq(F, L.reshape(-1, n), rcond=None)[0]
    return _param_metrics(com, coef[0], coef[1:].T, None)


def _param_metrics(com, r, A, h, zero_tol=1e-12):
    n, r_gt, A_gt = com['n_species'], com['r'], com['A']
    off = ~np.eye(n, dtype=bool)
    nz = off & (np.abs(A_gt) > zero_tol)
    m = dict(A_err=rel_err(A, A_gt), A_offdiag_err=rel_err(A[off], A_gt[off]), r_err=rel_err(r, r_gt),
             sign_acc=float(np.mean(np.sign(A[nz]) == np.sign(A_gt[nz]))))
    if h is not None:
        o = [i for i in range(n) if i != h]
        m.update(A_obs_block_err=rel_err(A[np.ix_(o, o)], A_gt[np.ix_(o, o)]), A_hidden_col_err=rel_err(A[o, h], A_gt[o, h]),
                 A_hidden_row_err=rel_err(A[h, o], A_gt[h, o]), r_obs_err=rel_err(r[o], r_gt[o]), r_hidden_err=rel_err(r[h], r_gt[h]))
    return m


def most_influential_species(com):
    """Species with the largest total outgoing interaction strength sum_i |A_ih| (i != h)."""
    A = np.abs(com['A']).copy()
    np.fill_diagonal(A, 0.0)
    return int(np.argmax(A.sum(axis=0)))
