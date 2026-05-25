import math


def normalise_float(value: float | None) -> float | None:
    if value is None:
        return 0.0

    if math.isnan(value):
        return None

    if math.isinf(value):
        value = 1e100 if value > 0 else -1e100
    return value
