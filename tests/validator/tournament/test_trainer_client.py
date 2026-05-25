from core.service_paths import GET_GPU_AVAILABILITY_ENDPOINT
from validator.tournament.trainer_client import trainer_url


def test_trainer_url_adds_default_port():
    assert trainer_url("10.0.0.5", GET_GPU_AVAILABILITY_ENDPOINT) == "http://10.0.0.5:8001/v1/trainer/get_gpu_availability"


def test_trainer_url_preserves_explicit_port():
    assert trainer_url("10.0.0.5:9000", GET_GPU_AVAILABILITY_ENDPOINT) == "http://10.0.0.5:9000/v1/trainer/get_gpu_availability"
