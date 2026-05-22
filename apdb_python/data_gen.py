from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    # Match MATLAB "twister" family by using MT19937 instead of PCG64 default.
    return np.random.Generator(np.random.MT19937(seed))


def _random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def generate_random_qcqp(n_constraints: int, m_vars: int, rng: np.random.Generator) -> dict:
    s = _random_orthogonal(rng, m_vars)
    dvals = rng.random(m_vars - 1) * 100.0
    a = s.T @ np.diag(np.concatenate([dvals, [1e-10]])) @ s
    a = 0.5 * (a + a.T)

    b = rng.standard_normal(m_vars)
    eigs = np.linalg.eigvalsh(a)
    sc = float(np.min(eigs))
    if sc < 1e-8:
        sc = 0.0

    d = rng.standard_normal((n_constraints, m_vars))
    q_list = []
    for _ in range(n_constraints):
        s_q = _random_orthogonal(rng, m_vars)
        d_q = rng.random(m_vars - 1) * 100.0
        q_tmp = s_q.T @ np.diag(np.concatenate([d_q, [1e-10]])) @ s_q
        q_list.append(0.5 * (q_tmp + q_tmp.T))

    c = rng.random(n_constraints)
    return {"A": a, "Q": q_list, "b": b, "d": d, "c": c, "sc": sc}
