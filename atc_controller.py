

import numpy as np
import random
import math
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
    >>> _spawn_aircraft()

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
