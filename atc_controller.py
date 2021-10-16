

import math
import numpy as np
import pandas as pd
import random
from typing import Tuple

from pandas.core.construction import array


# Airspace constants.
CONTROL_ZONE_RADIUS = 3  # Km.
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


def _spawn_aircraft() -> Tuple[float, float, float, str]:
    """Return aircraft on control zone boundary.

    Returns
    -------
    Tuple[float, float, float, str]
        Return a tuple containing the x-coordinate, y-coordinate, initial
        heading, and initial state of the spawned aircraft. The coordinates
        will be in km whereas the heading will be in degrees clockwise of
        North.

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
    ang = ang if x < 0 else 2 * math.pi - ang

    return x, y, ang, "A"


def _initialize_aircraft_info(t_sim: float, ac_spawn_rate: int) -> pd.DataFrame:
    """Generate aircraft information database.

    Parameters
    ----------
    t_sim : float
        Simulation time in seconds.
    ac_spawn_rate : int
        Aircraft spawn rate in aircraft per second. Values less then or equal
        to one are acceptable.

    Returns
    -------
    pd.DataFrame
        Aircraft data base containing a column for each aircraft and row for
        each position reporting cycle.
    
    Notes
    -----
    The resolution of the data limits viable aircraft spawn rates. It is
    recommended to use spawn rates less then or equal to one.

    The columns will defult to zero for times that the aircraft is not
    initialized for or has not been propagated for. The data frame will also
    contain the inital positions at various time steps for aircraft added to
    suystem.
    """


    timesteps_num = t_sim * TIME_STEP_FREQUENCY
    index = np.arange(0, t_sim, 1 / TIME_STEP_FREQUENCY)

    aircraft_num = ac_spawn_rate * t_sim
    aircraft_arr = np.zeros((timesteps_num, aircraft_num), dtype=object)

    aircraft_info = pd.DataFrame(aircraft_arr, index=index)

    time_steps_per_ac = math.floor(timesteps_num / aircraft_num)
    for i in range(aircraft_num):

        ind = index[i * time_steps_per_ac]
        aircraft_info[i][ind] = _spawn_aircraft()

    return aircraft_info


def _spawn_runways() -> pd.DataFrame:
    """Return `DataFrame` with runway locator points.

    Returns
    -------
    pd.DataFrame
        Data base containing the runway locator points.

    Notes
    -----
    This function will generate the data base containing points centered at
    both thresholds of each runway spawned. The runways form a parallel array
    of runways centered at the origin of the control zone spanning lengthwise.

    The columns of the data base are x1-coordinate, y1-coordinate,
    x2-coordinate, y2-coordinate, and the runway status value.

    Examples
    --------
    With `NUMBER_OF_RUNWAYS=5`.

    >>> _spawn_runways()
         0     1    2     3    4
    0 -1.2 -0.25 -1.2  0.25  0.0
    1 -0.6 -0.25 -0.6  0.25  0.0
    2  0.0 -0.25  0.0  0.25  0.0
    3  0.6 -0.25  0.6  0.25  0.0
    4  1.2 -0.25  1.2  0.25  0.0
    """

    n = NUMBER_OF_RUNWAYS
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


def _spawn_holding_patterns() -> pd.DataFrame:
    """Return the locations of holding pattern centers.

    Returns
    -------
    pd.DataFrame
        Data base containing the holding pattern locator points.

    Notes
    -----
    This function calculates the number of risk-free holding patterns that can
    be added onto the circle of `radius` and determines the points.

    Examples
    --------
    With `CONTROL_ZONE_RADIUS=3`, `HOLDING_PATTERN_RADIUS=1`, and
    `MIN_SEPARATION=0.1`.

    >>> _spawn_holding_patterns()
              0         1    2
    0  0.000000  1.800000  0.0
    1  1.711902  0.556231  0.0
    2  1.058013 -1.456231  0.0
    3 -1.058013 -1.456231  0.0
    4 -1.711902  0.556231  0.0
    """

    # Determine the number of holding patterns too create.
    radius = CONTROL_ZONE_RADIUS - HOLDING_PATTERN_RADIUS - 2 * MIN_SEPARATION
    c = 2 * math.pi * radius
    M = math.floor(c / (2 * HOLDING_PATTERN_RADIUS - MIN_SEPARATION))

    holding_pattern_data = np.empty((M, 3))

    # Calculate the angle between successive holding patterns.
    m = np.arange(0, M, 1)
    theta = 2 * math.pi * m / M

    # Determine holding patter center points.
    holding_pattern_data[:, 0] = radius * np.sin(theta)
    holding_pattern_data[:, 1] = radius * np.cos(theta)
    holding_pattern_data[:, 2] = 0

    holding_pattern_info = pd.DataFrame(holding_pattern_data)
    return holding_pattern_info
