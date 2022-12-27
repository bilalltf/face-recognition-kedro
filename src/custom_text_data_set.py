from typing import Any, Dict
import numpy as np
from kedro.io import AbstractDataSet, DataSetError

class CustomTextDataSet(AbstractDataSet):
    def __init__(self, filepath: str, fmt: str):
        self._filepath = filepath
        self._fmt = fmt

    def _save(self, data: np.ndarray) -> None:
        np.savetxt(self._filepath, data, fmt=self._fmt)

    def _load(self) -> np.ndarray:
        return np.loadtxt(self._filepath, dtype=np.str)

    def _describe(self) -> dict:
        return dict(filepath=self._filepath, fmt=self._fmt)

