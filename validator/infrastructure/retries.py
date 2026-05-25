import httpx
from requests.exceptions import HTTPError
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from core.logging import get_logger


logger = get_logger(__name__)

from tenacity import retry_if_exception_type


def should_retry_model_loading_on_exception(e):
    ephemeral_error_patterns = [
        "does not appear to have a file named",
        "does not appear to have files named",
        "Too Many Requests for url",
    ]

    while e is not None:
        if isinstance(e, HTTPError):
            if e.response is None:
                logger.error(f"HTTPError with no response: {e}, cause: {e.__cause__}")
                return True
            elif 500 <= e.response.status_code < 600:
                return True
        if any(pattern in str(e) for pattern in ephemeral_error_patterns):
            return True
        e = e.__cause__
    return False


def retry_on_5xx():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.5, min=30, max=600),
        retry=retry_if_exception(should_retry_model_loading_on_exception),
        reraise=True,
    )


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
