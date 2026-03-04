import numpy as np


def _r(x, k=3):
    try:
        return f"{float(x):.{k}f}"
    except Exception:
        return str(x)


def _rn(arr, k=3):
    try:
        a = np.asarray(arr).ravel()
        return "(" + ",".join(f"{float(t):.{k}f}" for t in a) + ")"
    except Exception:
        return str(arr)
