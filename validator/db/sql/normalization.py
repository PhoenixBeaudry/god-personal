import math

import httpx
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.logging import get_logger


logger = get_logger(__name__)

retry_http_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)

retry_http_fast = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.1, max=1),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True,
)

def normalise_float(float: float | None) -> float | None:
    if float is None:
        return 0.0

    if math.isnan(float):
        return None

    if math.isinf(float):
        float = 1e100 if float > 0 else -1e100
    return float
