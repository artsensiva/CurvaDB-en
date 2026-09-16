"""Independent high-precision oracle for the continuous Frechet distance.

Deliberately NOT sharing code with traj.frechet_cont: re-derives the same
free-space-diagram algorithm from scratch using mpmath, so that an algebra or
indexing mistake in one implementation is unlikely to be mirrored in the
other. Test-only -- too slow for anything but small fixtures (n, m <= ~6).
"""

from __future__ import annotations

import mpmath


def _quadratic_interval(ax, ay, bx, by, px, py, eps2, dps):
    """Interval of t in [0,1] with |a + t*(b-a) - p|^2 <= eps2. (lo, hi), lo > hi if empty."""
    with mpmath.workdps(dps):
        ax, ay, bx, by, px, py, eps2 = (mpmath.mpf(v) for v in (ax, ay, bx, by, px, py, eps2))
        dx, dy = bx - ax, by - ay
        coeff_quad = dx * dx + dy * dy
        gx, gy = ax - px, ay - py
        if coeff_quad < mpmath.mpf("1e-40"):
            inside = (gx * gx + gy * gy) <= eps2
            return (mpmath.mpf(0), mpmath.mpf(1)) if inside else (mpmath.mpf(1), mpmath.mpf(0))

        coeff_lin = 2 * (dx * gx + dy * gy)
        coeff_const = gx * gx + gy * gy - eps2
        discriminant = coeff_lin * coeff_lin - 4 * coeff_quad * coeff_const
        if discriminant < 0:
            return (mpmath.mpf(1), mpmath.mpf(0))

        root_delta = mpmath.sqrt(discriminant)
        root_a = (-coeff_lin - root_delta) / (2 * coeff_quad)
        root_b = (-coeff_lin + root_delta) / (2 * coeff_quad)
        t_lo = root_a if root_a > 0 else mpmath.mpf(0)
        t_hi = root_b if root_b < 1 else mpmath.mpf(1)
        if t_lo > t_hi:
            return (mpmath.mpf(1), mpmath.mpf(0))
        return (t_lo, t_hi)


def decide_mp(P, Q, eps, dps=50) -> bool:
    """Alt-Godau reachability, independently re-derived with mpmath arithmetic."""
    with mpmath.workdps(dps):
        n = len(P) - 1
        m = len(Q) - 1
        eps2 = mpmath.mpf(eps) * mpmath.mpf(eps)

        def feasible(a, b):
            dx = mpmath.mpf(a[0]) - mpmath.mpf(b[0])
            dy = mpmath.mpf(a[1]) - mpmath.mpf(b[1])
            return dx * dx + dy * dy <= eps2

        if not feasible(P[0], Q[0]) or not feasible(P[n], Q[m]):
            return False

        left = [[None] * m for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(m):
                left[i][j] = _quadratic_interval(
                    Q[j][0], Q[j][1], Q[j + 1][0], Q[j + 1][1], P[i][0], P[i][1], eps2, dps
                )

        bot = [[None] * (m + 1) for _ in range(n)]
        for i in range(n):
            for j in range(m + 1):
                bot[i][j] = _quadratic_interval(
                    P[i][0], P[i][1], P[i + 1][0], P[i + 1][1], Q[j][0], Q[j][1], eps2, dps
                )

        l_reach = [[None] * m for _ in range(n + 1)]
        b_reach = [[None] * (m + 1) for _ in range(n)]

        def ok(reach_row, j):
            return reach_row[j] is not None

        for i in range(n + 1):
            for j in range(m):
                lo, hi = left[i][j]
                if lo > hi:
                    continue
                entire = i == 0 and j == 0
                if not entire and i >= 1 and ok(b_reach[i - 1], j):
                    entire = True
                if not entire and j >= 1 and ok(l_reach[i], j - 1) and left[i][j - 1][1] == 1 and lo == 0:
                    entire = True
                if entire:
                    l_reach[i][j] = lo
                elif i >= 1 and ok(l_reach[i - 1], j):
                    new_lo = max(l_reach[i - 1][j], lo)
                    if new_lo <= hi:
                        l_reach[i][j] = new_lo

            if i >= n:
                continue
            for j in range(m + 1):
                lo, hi = bot[i][j]
                if lo > hi:
                    continue
                entire = i == 0 and j == 0
                if not entire and j >= 1 and ok(l_reach[i], j - 1):
                    entire = True
                if not entire and i >= 1 and ok(b_reach[i - 1], j) and bot[i - 1][j][1] == 1 and lo == 0:
                    entire = True
                if entire:
                    b_reach[i][j] = lo
                elif j >= 1 and ok(b_reach[i], j - 1):
                    new_lo = max(b_reach[i][j - 1], lo)
                    if new_lo <= hi:
                        b_reach[i][j] = new_lo

        return ok(l_reach[n], m - 1) or ok(b_reach[n - 1], m)


def distance_mp(P, Q, tol=mpmath.mpf("1e-9"), dps=50):
    """Continuous Frechet distance via bisection on decide_mp; returns the hi bracket."""
    with mpmath.workdps(dps):
        tol = mpmath.mpf(tol)
        if decide_mp(P, Q, 0.0, dps=dps):
            return mpmath.mpf(0)

        def dist(a, b):
            return mpmath.sqrt((mpmath.mpf(a[0]) - mpmath.mpf(b[0])) ** 2 + (mpmath.mpf(a[1]) - mpmath.mpf(b[1])) ** 2)

        hi = max(dist(P[0], Q[0]), dist(P[-1], Q[-1]))
        if hi <= 0:
            hi = tol
        while not decide_mp(P, Q, hi, dps=dps):
            hi *= 2

        lo = mpmath.mpf(0)
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if decide_mp(P, Q, mid, dps=dps):
                hi = mid
            else:
                lo = mid
        return hi
