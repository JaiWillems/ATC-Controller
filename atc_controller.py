

import math
import numpy as np
import pandas as pd
import random
from typing import Tuple


# Airspace constants.
CONTROL_ZONE_RADIUS = 10  # Km.
HOLDING_PATTERN_RADIUS = 1  # Km.


# Runway Specifications.
NUMBER_OF_RUNWAYS = 2  # Int.
RUNWAY_SEPARATION = 0.5  # Km.
RUNWAY_LENGTH = 0.5  # Km.
RUNWAY_WIDTH = 0.1  # Km.


# Aircraft constants.
MAX_SPEED = 0.14  # km/s.
MIN_SEPARATION = 0.1  # Km.


# Other.
TIME_STEP_FREQUENCY = 10  # Iterations per second.


def _spawn_aircraft() -> Tuple[float, float, float]:
    """Return aircraft on control zone boundary.

    Returns
    -------
    Tuple[float, float, float]
        Return a tuple containing the x-coordinate, y-coordinate, and initial
        heading of the spawned aircraft. The coordinates will be in km whereas
        the heading will be in degrees clockwise of North.

    Notes
    -----
    This function will randomly select an x-coordinate within the domain of the
    control zone and the sign of the y-coordinate. Using the equation of a
    circle with the the radius equal to the control zone radius, a y-coordinate
    is found such that the (x, y) pair lies on the control zone boundary.

    Examples
    --------
    With `CONTROL_ZONE_RADIUS = 10`, an example aircraft spawn event is as
    follows:

    >>> _spawn_aircraft()
    (5.486794024557362, 8.360328422501214, 3.7223764711048473)
    """

    # Get aircraft coordinates.
    x = random.uniform(-CONTROL_ZONE_RADIUS, CONTROL_ZONE_RADIUS)
    y = math.sqrt(CONTROL_ZONE_RADIUS ** 2 - x ** 2)
    y = y if random.randint(0, 1) else -y
    ac_position = np.array([x, y])

    # Get aircraft heading.
    num = np.sum(-ac_position * np.array([0, 1]))
    denom = np.linalg.norm(ac_position)
    ang = np.arccos(num / denom)
    print(ang)
    ang = ang if x < 0 else 2 * math.pi - ang

    return x, y, ang


def _spawn_runways(n: int) -> pd.DataFrame:
    """Return `DataFrame` with runway locator points.

    Parameters
    ----------
    n : int
        Number of runways to spawn.

    Returns
    -------
    pd.DataFrame
        Data base containing the runway locator points.

    Notes
    -----
    This function will generate the data base containing points centered at
    both thresholds of each runway spawned. The runways form a parallel array
    of runways centered at the origin of the control zone spanning lengthwise.
    """

    runway_data = np.empty((n, 5))

    if not n % 2:
        for i, N in enumerate(range(1, n, 2)):

            x = N * (RUNWAY_SEPARATION + RUNWAY_WIDTH) / 2
            y_base, y_top = - RUNWAY_LENGTH / 2, RUNWAY_LENGTH / 2

            runway_data[i, 0] = x
            runway_data[i, 1] = y_base
            runway_data[i, 2] = x
            runway_data[i, 3] = y_top
            runway_data[i, 4] = 0

            runway_data[i + n // 2, 0] = - x
            runway_data[i + n // 2, 1] = y_base
            runway_data[i + n // 2, 2] = - x
            runway_data[i + n // 2, 3] = y_top
            runway_data[i + n // 2, 4] = 0

    else:
        for i, N in enumerate(range(- n // 2 + 1, n // 2 + 1)):

            x = N * (RUNWAY_SEPARATION + RUNWAY_WIDTH)
            y_base, y_top = - RUNWAY_LENGTH / 2, RUNWAY_LENGTH / 2

            runway_data[i, 0] = x
            runway_data[i, 1] = y_base
            runway_data[i, 2] = x
            runway_data[i, 3] = y_top
            runway_data[i, 4] = 0

    runway_info = pd.DataFrame(runway_data)
    return runway_info
