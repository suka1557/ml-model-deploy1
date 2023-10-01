import os
import pandas as pd
from typing import Union
from pydantic import BaseModel

class ScaleInputs(BaseModel):
    MAX_VALUE: float = 255.0

    def scale_input(self, input_df:pd.DataFrame) -> pd.DataFrame:
        return input_df/self.MAX_VALUE
