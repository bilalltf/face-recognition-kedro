# src/custom_text_data_set.py

from typing import Any, Dict
import numpy as np
from kedro.io import AbstractDataSet, DataSetError

class CustomTextDataSet(AbstractDataSet):
    def __init__(self, filepath: str, fmt: str = "%.18e", **kwargs):
        super().__init__(filepath=filepath, **kwargs)
        self._fmt = fmt
    
    def _save(self, data: Any) -> None:
        np.savetxt(self._filepath, data, fmt=self._fmt)
    
    def _load(self) -> Any:
        # your custom code to load data from the file at self._filepath
        pass
    
    def _exists(self) -> bool:
        # your custom code to check if the file at self._filepath exists
        pass
    
    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
