"""Szenario: Zufallsspiele, Startzahlen, Rollenmatrizen, Normalzahlen; SplitMix64-Kopie und Referenzwerte."""

import ast
from pathlib import Path

import numpy as np
import pytest

import kos_scenario as S
from kos_rng import SplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "kos_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert 0xE220A8397B1DCDAF == 16294208416658607535          # dieselbe Zahl, wie sie lp-mitte-demo dezimal führt


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "kos_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_draw_games_are_deterministic_and_in_range():
    a1, b1, c1, d1 = S.draw_games(7, 1000)
    a2, *_ = S.draw_games(7, 1000)
    assert np.array_equal(a1, a2) and len(a1) == 1000
    for v in (a1, b1, c1, d1):
        assert v.min() >= 0 and v.max() < 10
    assert not np.array_equal(a1, S.draw_games(8, 1000)[0])


def test_draw_games_first_value_is_the_first_uniform_times_ten():
    a, b, c, d = S.draw_games(7, 3)
    u = uniform_block(7, 12) * 10
    assert (a[0], b[0], c[0], d[0]) == (u[0], u[1], u[2], u[3]) and a[1] == u[4]


def test_draw_games_unknown_mode_raises():
    with pytest.raises(ValueError):
        S.draw_games(1, 10, "irgendwas")


def test_coordination_subset_conditions():
    a = np.array([5.0, 5.0, 2.0, 5.0]); b = np.array([2.0, 5.0, 2.0, 2.0]); c = np.array([1.0, 1.0, 1.0, 3.0]); d = np.array([2.0, 2.0, 2.0, 2.0])
    # Spiel 0: a>d, b>c, a!=b ok; Spiel 1: a = b ausgeschlossen; Spiel 2: a = d ausgeschlossen; Spiel 3: b<c ausgeschlossen
    assert S.coordination_subset(a, b, c, d).tolist() == [True, False, False, False]


def test_start_count_boundary_is_exact():
    """Start 50 %: eine Zufallszahl u zählt als S, wenn int(u·10^6) < 500 000, also u < 0,5. Mittel über viele Spieler nahe der Hälfte."""
    rng = SplitMix64(11)
    counts = [S.start_count(100, 50, rng) for _ in range(200)]
    assert abs(np.mean(counts) - 50) < 1.5
    assert S.start_count(10, 0, SplitMix64(1)) == 0 and S.start_count(10, 100, SplitMix64(1)) == 10


def test_start_count_uses_one_number_per_player():
    r1, r2 = SplitMix64(5), SplitMix64(5)
    S.start_count(7, 50, r1)
    for _ in range(7):
        r2.uniform()
    assert r1.state == r2.state


def test_role_matrices():
    A, B = S.role_matrices(3, 2)
    assert A == [[3, 0], [0, 1]] and B == [[1, 0], [0, 2]]


def test_threshold_normals_shape_and_determinism():
    z = S.threshold_normals(3, 4, 5)
    assert z.shape == (4, 5) and np.array_equal(z, S.threshold_normals(3, 4, 5))
