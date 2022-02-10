from typing import Any, Union

import pandas as pd
from numpy import typing as nptyping

ArrayLike = Union[pd.DataFrame, pd.Series, nptyping.NDArray[Any]]
