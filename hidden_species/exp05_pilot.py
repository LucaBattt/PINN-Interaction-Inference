from runner import run_jobs
L = dict(lbfgs_iters=5000)
jobs = [
    dict(tag="n3 s1 h=1 3pts", community=dict(n_species=3, seed=1), train=dict(hidden=1, n_hidden_pts=3, **L)),
    dict(tag="n3 s1 h=0 1pt", community=dict(n_species=3, seed=1), train=dict(hidden=0, n_hidden_pts=1, **L)),
    dict(tag="n4 s4 h=3 3pts", community=dict(n_species=4, seed=4), train=dict(hidden=3, n_hidden_pts=3, **L)),
    dict(tag="n4 s4 h=3 3pts varpro", community=dict(n_species=4, seed=4), train=dict(hidden=3, n_hidden_pts=3, varpro=True, **L)),
    dict(tag="n5 s18 h=2 3pts", community=dict(n_species=5, seed=18), train=dict(hidden=2, n_hidden_pts=3, **L)),
    dict(tag="n5 s18 h=2 3pts varpro", community=dict(n_species=5, seed=18), train=dict(hidden=2, n_hidden_pts=3, varpro=True, **L)),
    dict(tag="n5 s18 fully observed", community=dict(n_species=5, seed=18), train=dict()),
    dict(tag="n5 s18 h=2 1pt", community=dict(n_species=5, seed=18), train=dict(hidden=2, n_hidden_pts=1, **L)),
]
if __name__ == '__main__':
    run_jobs(jobs, 'results/exp05.jsonl')
