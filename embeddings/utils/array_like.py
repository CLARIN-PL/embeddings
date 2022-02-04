from typing import Any, Union

from numpy import typing as nptyping
import pandas as pd


ArrayLike = Union[pd.DataFrame, pd.Series, nptyping.NDArray[Any]]
