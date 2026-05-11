# causal_discovery_simple.py
import itertools


def _powerset(s):
    s = list(s)
    for r in range(len(s) + 1):
        for comb in itertools.combinations(s, r):
            yield set(comb)


def causal_discovery(DI, IS, subgoals, tau=0.05, min_support=30):
    """
    Return parents_hat: dict[child] = set(parents).
    Parent nodes are restricted to IS, because only IS nodes are controllable in the current iteration.

    This version is robust to missing keys in DI records, although sampling.py should normally
    write all canonical keys.
    """
    IS = set(IS)
    parents_hat = {g: set() for g in subgoals}

    rows = []
    for e in DI:
        vars_after = e.get("vars_after_int", {})
        vars_max = e.get("vars_max_window", {})
        cond = {g for g in IS if int(vars_after.get(g, 0)) == 1}
        y = {g: int(vars_max.get(g, 0)) for g in subgoals}
        rows.append((cond, y))

    for child in subgoals:
        if child in IS:
            continue

        best_P = None
        best_score = 0.0

        for P in _powerset(IS):
            if len(P) == 0:
                continue

            y1, y0 = [], []
            for cond, y in rows:
                if P.issubset(cond):
                    y1.append(y[child])
                else:
                    y0.append(y[child])

            if len(y1) < min_support or len(y0) < min_support:
                continue

            p1 = sum(y1) / float(len(y1))
            p0 = sum(y0) / float(len(y0))
            score = p1 - p0

            if score >= tau:
                if (best_P is None) or (len(P) < len(best_P)) or (len(P) == len(best_P) and score > best_score):
                    best_P = set(P)
                    best_score = score

        if best_P is not None:
            parents_hat[child] = best_P

    return parents_hat
