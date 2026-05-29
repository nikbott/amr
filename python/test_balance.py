"""
Regression tests for the linear-tree refine/balance reference implementation.

Numba-free on purpose: this is the parity oracle for the C++ backends, so it
must run anywhere (CI without a GPU). Run with:  python3 -m unittest test_balance
"""
import unittest

from tree import Quadtree, Octree
from morton import Morton2D, Morton3D


def _boxes(tree):
    out = []
    for n in tree.leaves:
        coords = Morton2D.decode(n.code) if tree.ndim == 2 else Morton3D.decode(n.code)
        size = 1 << (tree.max_level - n.level)
        out.append((coords, size, n.level))
    return out


def _face_adjacent(a, b, ndim):
    """True iff cells a and b share exactly one (ndim-1)-dimensional face of positive extent."""
    (ca, sa, _), (cb, sb, _) = a, b
    shared_planes = 0
    for k in range(ndim):
        lo_a, hi_a = ca[k], ca[k] + sa
        lo_b, hi_b = cb[k], cb[k] + sb
        if hi_a == lo_b or hi_b == lo_a:
            shared_planes += 1
        elif max(lo_a, lo_b) < min(hi_a, hi_b):
            continue
        else:
            return False
    return shared_planes == 1


def _max_face_level_jump(tree):
    """Largest |level difference| over all face-adjacent leaf pairs (gold-standard 2:1 check)."""
    bx = _boxes(tree)
    worst = 0
    for i in range(len(bx)):
        for j in range(i + 1, len(bx)):
            if _face_adjacent(bx[i], bx[j], tree.ndim):
                worst = max(worst, abs(bx[i][2] - bx[j][2]))
    return worst


def _build(cls, max_level, oracle, steps):
    t = cls(max_level=max_level)
    for _ in range(steps):
        if not t.refine(oracle):
            break
    return t


# Oracles cap at max_level (refine trusts the oracle to stop; see refine()).
ORACLES = [
    ("corner",     lambda n, ml: n.code < (1 << (2 * (ml - 1))) and n.level < ml),
    ("lowlevel",   lambda n, ml: n.level < 3),
    ("stripe",     lambda n, ml: (n.code % 7) < 3 and n.level < 4),
    ("deepcorner", lambda n, ml: n.code == 0 and n.level < ml),
]


class TestMorton(unittest.TestCase):
    def test_roundtrip_2d(self):
        for xy in [(0, 0), (13, 27), (1023, 777)]:
            self.assertEqual(Morton2D.decode(Morton2D.encode(xy)), xy)

    def test_roundtrip_3d(self):
        for xyz in [(0, 0, 0), (5, 9, 17), (511, 300, 123)]:
            self.assertEqual(Morton3D.decode(Morton3D.encode(xyz)), xyz)


class TestBalance(unittest.TestCase):
    def test_invariants(self):
        for cls, ml in [(Quadtree, 8), (Octree, 6)]:
            for name, oracle in ORACLES:
                for steps in (3, 5, 7, 9):
                    with self.subTest(tree=cls.__name__, oracle=name, steps=steps):
                        t = _build(cls, ml, oracle, steps)
                        t.balance()
                        codes = [n.code for n in t.leaves]
                        self.assertEqual(codes, sorted(codes), "leaves not sorted")
                        self.assertEqual(len(codes), len(set(codes)), "duplicate codes")
                        self.assertLessEqual(
                            _max_face_level_jump(t), 1,
                            "2:1 balance violated (face-adjacent cells differ by >1 level)")

    def test_balance_is_fixed_point(self):
        # A balanced mesh must not change when balanced again.
        t = _build(Quadtree, 8, ORACLES[0][1], 6)
        t.balance()
        before = [(n.code, n.level) for n in t.leaves]
        t.balance()
        after = [(n.code, n.level) for n in t.leaves]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
