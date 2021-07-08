import time
from typing import ContextManager


class _MeasureManager:
    def __init__(self, measurement_name: str, result_dict: dict):
        self.measurement_name = measurement_name
        self.result_dict = result_dict

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result_dict[self.measurement_name] = time.perf_counter() - self.start


class Timer:
    def __init__(self):
        self.measurements = {}

    def measure(self, measurement_name: str) -> ContextManager:
        return _MeasureManager(measurement_name, self.measurements)
