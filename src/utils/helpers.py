import os
from typing import Tuple, Union

import numpy as np
from tensorflow import Tensor

from src.utils.constants import IndexableConstants as idc


def interpolate_inf(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inf_mask = np.isinf(vector)
    interpolations = np.interp(inf_mask.nonzero()[0],
                               ((~inf_mask).nonzero()[0]),
                               vector[~inf_mask])
    vector[inf_mask] = interpolations

    return vector.reshape(-1, 1), inf_mask.reshape(-1, 1)


def to_wider_limits(matrix: Tensor) -> Tensor:
    return (matrix - 0.5) * 2


def to_narrow_limits(matrix: Tensor) -> Tensor:
    return (matrix / 2) + 0.5


def to_singer_index(datapoint: str) -> int:
    return idc.SINGERS.index(os.path.basename(datapoint).split('-')[0])
