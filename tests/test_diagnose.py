"""Tests for the graph generator, simulator, inference methods and baselines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diagnose.baselines import most_alarmed_descendants, most_upstream, random_ranking  # noqa: E402
from diagnose.exact import exact_posterior  # noqa: E402
from diagnose.graph import chain_graph, generate_graph  # noqa: E402
from diagnose.metrics import total_variation  # noqa: E402
from diagnose.model import Params  # noqa: E402
from diagnose.sampling import likelihood_weighting  # noqa: E402
from diagnose.simulate import simulate_incident  # noqa: E402


def test_generator_reproducible() -> None:
    a, b = generate_graph(12, seed=7), generate_graph(12, seed=7)
    assert a == b
    assert a != generate_graph(12, seed=8)
    for i, ps in enumerate(a.parents):
        assert len(ps) <= 3
        assert all(j < i for j in ps)  # topological order


def test_simulator_reproducible() -> None:
    g = generate_graph(12, seed=3)
    x = simulate_incident(g, Params(), np.random.default_rng(42))
    y = simulate_incident(g, Params(), np.random.default_rng(42))
    assert x.root == y.root
    assert np.array_equal(x.failures, y.failures)
    assert np.array_equal(x.alarms, y.alarms)
    assert x.failures[x.root]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_posterior_sums_to_one(seed: int) -> None:
    g = generate_graph(10, seed=seed)
    inc = simulate_incident(g, Params(), np.random.default_rng(seed))
    post = exact_posterior(g, Params(), inc.alarms)
    assert post.shape == (10,)
    assert np.all(post >= 0)
    assert post.sum() == pytest.approx(1.0)


def test_chain_without_noise_blames_most_upstream_alarm() -> None:
    g = chain_graph(6)
    params = Params(p_prop=1.0, leak=0.0, fp_rate=0.0, fn_rate=0.0)
    alarms = np.array([0, 0, 1, 1, 1, 1], dtype=bool)  # root must be service 2
    post = exact_posterior(g, params, alarms)
    assert int(np.argmax(post)) == 2
    assert post[2] == pytest.approx(1.0)


def test_likelihood_weighting_close_to_exact() -> None:
    g = generate_graph(8, seed=5)
    params = Params()
    inc = simulate_incident(g, params, np.random.default_rng(5))
    exact = exact_posterior(g, params, inc.alarms)
    lw = likelihood_weighting(g, params, inc.alarms, 50_000, np.random.default_rng(0))
    assert total_variation(exact, lw.posterior) < 0.05
    assert 0 < lw.ess <= 50_000


def test_baselines_return_valid_rankings() -> None:
    g = generate_graph(12, seed=11)
    inc = simulate_incident(g, Params(), np.random.default_rng(11))
    rankings = [
        most_upstream(g, inc.alarms),
        most_alarmed_descendants(g, inc.alarms),
        random_ranking(g, np.random.default_rng(0)),
    ]
    for r in rankings:
        assert sorted(r) == list(range(g.n))


def test_run_all_quick(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "run_all.py"), "--quick", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    for name in ("accuracy_vs_noise", "sampling_convergence", "scaling", "failure_case"):
        assert (tmp_path / f"{name}.csv").exists()
        assert (tmp_path / f"{name}.png").exists()
    assert (tmp_path / "failure_case.txt").exists()
