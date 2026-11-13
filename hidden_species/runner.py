"""Runs batches of hidden-species experiments in parallel and appends the results to a JSON-lines file."""
import json, os, sys, time
import numpy as np
import multiprocessing as mp


def _run(job):
    import torch
    torch.set_num_threads(1)
    import hs_pinn as hs
    com = hs.make_community(**job['community'])
    res = hs.train(com, **job['train'])
    out = dict(tag=job.get('tag', ''), variant=job.get('variant', ''), community=job['community'], train=job['train'], metrics=res['metrics'],
               r=res['r'].tolist(), A=res['A'].tolist(), alpha=res['alpha'])
    if job.get('keep_traj'):
        out['x_pred'] = res['x_pred'].tolist()
    return out


def run_jobs(jobs, out_path, n_proc=4):
    t0 = time.time()
    results = []
    with mp.get_context('spawn').Pool(n_proc) as pool:
        for i, out in enumerate(pool.imap_unordered(_run, jobs)):
            results.append(out)
            with open(out_path, 'a') as f:
                f.write(json.dumps(out) + '\n')
            m = out['metrics']
            keys = ['A_err', 'A_obs_block_err', 'A_hidden_col_err', 'A_hidden_row_err', 'x_hidden_err', 'sign_acc', 'sim_obs_err']
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:5.0f}s] {out['tag']:<50s} " +
                  ' '.join(f"{k.replace('A_', '').replace('_err', '')}={m[k]:.3f}" for k in keys if k in m) + f" Ld={m['loss_data']:.1e} Lp={m['loss_phys']:.1e}", flush=True)
    return results


def load(path):
    return [json.loads(l) for l in open(path)]
