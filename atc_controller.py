import math
import numpy as np
import pandas as pd
import random
from typing import Tuple

from pandas.core.construction import array


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
POSIITION_TOLERANCE = 0.1  # Km.
CURR_LANDING_AC = 0  # Aircraft identifier.


def _get_ac_heading(x: float, y: float) -> float:
    """Return aircraft heading.

    Parameters
    ----------
    x : float
        Aicraft x-coordinate in Km.
    y : float
        Aircraft y-coordinate in Km.

    Returns
    -------
    float
        Return aircraft heading in radians.
    """

    ac_position = np.array([x, y])
    num = np.sum(-ac_position * np.array([0, 1]))
    denom = np.linalg.norm(ac_position)
    ang = np.arccos(num / denom)
    ang = ang if x < 0 else 2 * math.pi - ang

    return ang


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

    ang = _get_ac_heading(x, y)

    return x, y, ang, "A"


def _initialize_aircraft_info(t_sim: float, ac_spawn_rate: float) -> pd.DataFrame:
    """Generate aircraft information database.

    Parameters
    ----------
    t_sim : float
        Simulation time in seconds.
    ac_spawn_rate : float
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
    index = np.round(index, 1)

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


def _get_closest_control_zone(x: float, y: float, hp_info: pd.DataFrame) -> int:
    """Return index for the closest holding pattern.

    Parameters
    ----------
    x : float
        Aircraft x-coordinate.
    y : float
        Aircraft y-coordinate.
    hp_info : pd.DataFrame
        Holding pattern information.

    Returns
    -------
    int
        `hp_info` index for the closest holding pattern.
    """

    min_dist = CONTROL_ZONE_RADIUS
    min_ind = 0

    for ind in hp_info.index:
        hp_x = hp_info[0][ind]
        hp_y = hp_info[1][ind]

        dist = np.sqrt((x - hp_x) ** 2 + (y - hp_y) ** 2)

        if dist < min_dist:
            min_dist = dist
            min_ind = ind

    return min_ind


def _get_closest_threshold(x: float, y: float, rw_info: pd.DataFrame) -> Tuple[int, int, int]:
    """Return index for the closest runway threshold.

    Parameters
    ----------
    x : float
        Aircraft x-coordinate.
    y : float
        Aircraft y-coordinate.
    rw_info : pd.DataFrame
        Runway threshold information.

    Returns
    -------
    Tuple[int, int, int]
        `rw_info` index for the closest runway threshold. The first value is
        the row index and the latter two are the x and y position column
        indices.
    """

    min_dist = CONTROL_ZONE_RADIUS
    min_ind = (0, 0, 0)

    for ind in rw_info.index:

        hp_x = rw_info[0][ind]
        hp_y = rw_info[1][ind]

        dist1 = np.sqrt((x - hp_x) ** 2 + (y - hp_y) ** 2)

        hp_x = rw_info[2][ind]
        hp_y = rw_info[3][ind]

        dist2 = np.sqrt((x - hp_x) ** 2 + (y - hp_y) ** 2)

        if dist1 < min_dist:
            min_dist = dist1
            min_ind = (ind, 0, 1)
        elif dist2 < min_dist:
            min_dist = dist2
            min_ind = (ind, 3, 4)

    return min_ind


def _get_next_position(x: float, y: float, heading: float, state: str, hp_info:
                       pd.DataFrame, rw_info: pd.DataFrame, ac: int,
                       CURR_LANDING_AC) -> Tuple[float, float, float, str]:
    """Calculate the aircrafts next position.

    Parameters
    ----------
    x : float
        Aircraft x-coordinate in Km.
    y : float
        Aircraft y-coordinate in Km.
    heading : float
        Aircraft heading in decimal radians.
    state : str
        Aircraft current state.
    hp_info : pd.DataFrame
        Holding pattern positions.
    rw_info : pd.DataFrame
        Runway position information.
    ac : int
        Aircraft identifier.
    CURR_LANDING_AC : [type]
        Current landing aircraft.

    Returns
    -------
    Tuple[float, float, float, str]
        Return the next x, y, heading, and state information.
    """

    if state == "A":

        radius = np.sqrt(x ** 2 + y ** 2)

        min_R = CONTROL_ZONE_RADIUS - MIN_SEPARATION - POSIITION_TOLERANCE
        max_R = CONTROL_ZONE_RADIUS - MIN_SEPARATION + POSIITION_TOLERANCE

        if (min_R < radius) | (radius < max_R):

            hp_ind = _get_closest_control_zone(x, y, hp_info)

            if hp_info[2][hp_ind] == 0:

                state_new = "C"
                heading_new = _get_ac_heading(hp_info[0][hp_ind] - x, hp_info[1][hp_ind] - y)

            else:

                state_new = "B"
                heading_new = (hp_info[2][hp_ind] + np.pi / 2) % (2 * np.pi)

        else:

            state_new = "A"
            heading_new = heading

        x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
        y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

    elif state == "B":

        hp_ind = _get_closest_control_zone(x, y, hp_info)

        if hp_info[2][hp_ind] == 0:

            state_new = "C"
            heading_new = _get_ac_heading(hp_info[0][hp_ind] - x, hp_info[1][hp_ind] - y)

        else:

            state_new = "B"
            heading_new = heading - MAX_SPEED / (TIME_STEP_FREQUENCY * (CONTROL_ZONE_RADIUS - MIN_SEPARATION))

        x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
        y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

    elif state == "C":

        hp_ind = _get_closest_control_zone(x, y, hp_info)
        dist = np.sqrt((hp_info[0][hp_ind] - x) ** 2 + (hp_info[1][hp_ind] - y) ** 2)

        if dist < POSIITION_TOLERANCE + 1:

            state_new = "D"
            heading_new = heading

            x_new = x
            y_new = y

        else:

            state_new = "C"
            heading_new = heading

            x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
            y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

    elif state == "D":

        if ac == CURR_LANDING_AC:

            row_ind, x_ind, y_ind = _get_closest_threshold(x, y, rw_info)

            state_new = "E"
            heading_new = _get_ac_heading(rw_info[x_ind][row_ind] - x, rw_info[y_ind][row_ind] - y)

            x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
            y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

        else:

            state_new = "D"
            heading_new = heading

            x_new = x
            y_new = y

    elif state == "E":

        row_ind, x_ind, y_ind = _get_closest_threshold(x, y, rw_info)
        dist = np.sqrt((rw_info[x_ind][row_ind] - x) ** 2 + (rw_info[y_ind][row_ind] - y) ** 2)

        if (dist < MIN_SEPARATION) | (CURR_LANDING_AC == ac):

            x_ind = 0 if x_ind == 2 else 2
            y_ind = 1 if y_ind == 3 else 3

            CURR_LANDING_AC += 1

            state_new = "F"
            heading_new = _get_ac_heading(rw_info[x_ind][row_ind] - x, rw_info[y_ind][row_ind] - y)

            x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
            y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

        else:

            state_new = "E"
            heading_new = heading

            x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
            y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

    elif state == "F":

        row_ind, x_ind, y_ind = _get_closest_threshold(x, y, rw_info)
        dist = np.sqrt((rw_info[x_ind][row_ind] - x) ** 2 + (rw_info[y_ind][row_ind] - y) ** 2)

        if abs(dist - RUNWAY_LENGTH / 2) < POSIITION_TOLERANCE:
            x_new, y_new, heading_new, state_new = -1, -1, -1, "END"

        else:

            state_new = "F"
            heading_new = heading

            x_new = x + MAX_SPEED * np.sin(heading_new) / TIME_STEP_FREQUENCY
            y_new = y + MAX_SPEED * np.cos(heading_new) / TIME_STEP_FREQUENCY

    else:

        x_new, y_new, heading_new, state_new = -1, -1, -1, "END"

    return x_new, y_new, heading_new, state_new


def _propagate_positions(hp_info: pd.DataFrame, rw_info: pd.DataFrame, ac_info: pd.DataFrame) -> pd.DataFrame:
    """Propagate aircraft positions.

    Parameters
    ----------
    hp_info : pd.DataFrame
        Holding pattern informmation.
    rw_info : pd.DataFrame
        Runway position information.
    ac_info : pd.DataFrame
        Aircraft position and state information.

    Returns
    -------
    pd.DataFrame
        Propagated aircraft position and state information.
    """

    for time in ac_info.index:
        for ac in ac_info.columns:

            tup = ac_info[ac][time]
            if isinstance(tup, tuple):
                x, y, heading, state = tup
                tup_new = _get_next_position(x, y, heading, state, hp_info, rw_info, ac, CURR_LANDING_AC)

                if np.round(time + 1 / TIME_STEP_FREQUENCY, 1) < ac_info.index.max():
                    ac_info[ac][np.round(time + 1 / TIME_STEP_FREQUENCY, 1)] = tup_new

    return ac_info


def simulate(t_sim: float, ac_spawn_rate: float) -> None:
    """Simulate autonomous ATC system.

    This function will simulate the autonomous ATC system using the program
    defined constants and the simulation parameters. The propagated position
    data for the aircraft are exported as a csv file.

    Parameters
    ----------
    t_sim : float
        Simulation time in seconds.
    ac_spawn_rate : float
        Aircraft spawn rate in aircraft per second. Values less then or equal
        to one are acceptable.
    """

    hp_info = _spawn_holding_patterns()
    rw_info = _spawn_runways()
    ac_info = _initialize_aircraft_info(t_sim, ac_spawn_rate)

    propagated_data = _propagate_positions(hp_info, rw_info, ac_info)
    propagated_data.to_csv("propagated_simulation_data.csv")
