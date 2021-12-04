from typing import Tuple

import numpy as np


def interpolate_inf(vector: np.ndarray) -> Tuple[np.ndarray]:
    inf_mask = np.isinf(vector)
    interpolations = np.interp(inf_mask.nonzero()[0],
                               ((~inf_mask).nonzero()[0]),
                               vector[~inf_mask])
    vector[inf_mask] = interpolations

    return vector.reshape(-1, 1), inf_mask.reshape(-1, 1)