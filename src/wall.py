########################################################################################
# Imports
########################################################################################

import sys
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
WINDOW_SIZE = 8 # Window size to calculate the average distance
CURVE_ANGLE_SIZE = 40 # The angle of a gap to check existence of curve
CURVE_DISTANCE_SIZE = 30 # The distance of a gap to check existence of curve
FAR_DISTANCE_SIZE = 100 # If the closest point is over this value, it would be ignored

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

# Initialize PID control variables for angle
KP = 0.05  # Proportional constant for angle
KI = 0.0001  # Integral constant for angle
KD = 0.05  # Derivative constant for angle
prev_error_angle = 0  # Previous error for angle control
integral_angle = 0  # Integral term for angle control

# Initialize PID control variables for speed
KP_speed = 0.1  # Proportional constant for speed
KI_speed = 0.001  # Integral constant for speed
KD_speed = 0.05  # Derivative constant for speed
prev_error_speed = 0  # Previous error for speed control
integral_speed = 0  # Integral term for speed control

# Initialize desired speed
desired_speed = 1.0  # Set desired speed to 0.5 (you can adjust this value)

########################################################################################
# Functions
########################################################################################

def get_lidar_average_distance(
    scan: NDArray[Any, np.float32], angle: float, window_angle: float = 4
) -> float:
    """
    Finds the average distance of the object at a particular angle relative to the car.

    Args:
        scan: The samples from a LIDAR scan
        angle: The angle (in degrees) at which to measure distance, starting at 0
            directly in front of the car and increasing clockwise.
        window_angle: The number of degrees to consider around angle.

    Returns:
        The average distance of the points at angle in cm.

    Note:
        Ignores any samples with a value of 0.0 (no data).
        Increasing window_angle reduces noise at the cost of reduced accuracy.
    """

    start_offset = int(angle - window_angle / 2) * 2 % 720
    end_offset = int(angle + window_angle / 2) * 2 % 720

    range_values = scan[start_offset:end_offset]
    if start_offset > end_offset: 
        range_values = np.concatenate((scan[start_offset:], scan[:end_offset]))

    valid_values = range_values[range_values > 0]
    if valid_values.size == 0:
        return 0.0

    return float(np.mean(valid_values))

def get_lidar_closest_point(
    scan: NDArray[Any, np.float32], window: Tuple[float, float] = (0, 360)
) -> Tuple[float, float]:
    """
    Finds the closest point from a LIDAR scan.

    Args:
        scan: The samples from a LIDAR scan.
        window: The degree range to consider, expressed as (min_degree, max_degree)

    Returns:
        The (angle, distance) of the point closest to the car within the specified
        degree window. All angles are in degrees, starting at 0 directly in front of the
        car and increasing clockwise. Distance is in cm.

    Note:
        Ignores any samples with a value of 0.0 (no data).

        In order to define a window which passes through the 360-0 degree boundary, it
        is acceptable for window min_degree to be larger than window max_degree.  For
        example, (350, 10) is a 20 degree window in front of the car.
    """

    converted_scan = (scan - 0.01) % 10000

    start_offset = int(window[0] % 360)
    end_offset = int(window[1] % 360)

    if start_offset < end_offset:
        converted_scan = converted_scan[start_offset:end_offset]
    else:
        converted_scan = np.concatenate((converted_scan[start_offset:], converted_scan[:end_offset]))

    angle = np.argmin(converted_scan)
    real_angle = (start_offset + angle) % 360

    return (real_angle, scan[int(real_angle)])

def update_lidar():
    """
    Receive the lidar samples and get the average samples from it
    """
    global closest_left_angle
    global closest_left_distance
    global closest_right_angle
    global closest_right_distance
    global closest_front_angle
    global closest_front_distance
    global average_scan

    scan = rc.lidar.get_samples()
    average_scan = np.array([get_lidar_average_distance(scan, angle, WINDOW_SIZE) for angle in range(360)])

    closest_left_angle, closest_left_distance = get_lidar_closest_point(average_scan, (-90, -60))
    closest_right_angle, closest_right_distance = get_lidar_closest_point(average_scan, (60, 90))
    closest_front_angle, closest_front_distance = get_lidar_closest_point(average_scan, (-30, 30))
    # TODO
    # 30 to 60, -30 to -60 are blind spots. However, this setting makes the movement smoother. 
    # For blind spots, it would be better to find the closest point separately and handle it.
    return

def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)
    # Set update_slow to refresh every half second
    #rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Wall Following\n"
        "\n"
        "Controls:\n"
        "    A button = print current speed, angle, and closest values\n"
    )

def update():
    global speed
    global angle
    global prev_error_angle
    global prev_error_speed
    global integral_angle
    global integral_speed
    global closest_left_angle
    global closest_left_distance
    global closest_right_angle
    global closest_right_distance
    global closest_front_angle
    global closest_front_distance
    global average_scan

    update_lidar()

    # If there are no obstacles on either side, just run straight.
    if (closest_left_distance > FAR_DISTANCE_SIZE or closest_left_distance == 0) and \
        (closest_right_distance > FAR_DISTANCE_SIZE or closest_right_distance == 0):
            angle = 0
    else:
        angle_error = 0
        left_gap = average_scan[270 + CURVE_ANGLE_SIZE] - average_scan[270]
        right_gap = average_scan[90 - CURVE_ANGLE_SIZE] - average_scan[90]

        if left_gap > CURVE_DISTANCE_SIZE and left_gap > right_gap: # Left Curve
            angle_error = -CURVE_DISTANCE_SIZE
        elif right_gap > CURVE_DISTANCE_SIZE: # Right Curve
            angle_error = CURVE_DISTANCE_SIZE
        else: # Try to maintain the center
            center = (closest_left_distance + closest_right_distance) / 2
            angle_error = (center - closest_left_distance) * 2

            # If an obstacle is expected to be in front of one side, move to a higher angle.
            if angle_error > 0:
                angle_error = (1 + (closest_left_angle - 270) / 30) * angle_error
            else:
                angle_error = (1 + (90 - closest_right_angle) / 30) * angle_error

        # Update angle integral term
        integral_angle += angle_error

        # Update angle derivative term
        angle_derivative = angle_error - prev_error_angle
        prev_error_angle = angle_error

        # Calculate angle PID output
        angle_pid_output = KP * angle_error + KI * integral_angle + KD * angle_derivative

        # Convert angle PID output to angle
        angle = angle_pid_output

    # PID control for speed
    # Calculate speed error (difference between desired and actual speed)
    speed_error = desired_speed - speed

    # If a crash is expected, slow down
    closest_side_distance = min(closest_left_distance, closest_right_distance)
    if closest_front_distance < closest_side_distance * 2:
        speed_error -= 0.1
        # speed_error -= (1 * (closest_front_distance / (closest_side_distance * 4)))
        # TODO: Find a better formula..

    # Update speed integral term
    integral_speed += speed_error

    # Update speed derivative term
    speed_derivative = speed_error - prev_error_speed
    prev_error_speed = speed_error

    # Calculate speed PID output
    speed_pid_output = KP_speed * speed_error + KI_speed * integral_speed + KD_speed * speed_derivative

    # Convert speed PID output to speed
    speed += speed_pid_output

    # Constrain speed and angle within 0.0 to 1.0
    speed = max(0.0, min(1.0, speed))
    angle = max(-1.0, min(1.0, angle))

    # Set the speed and angle of the car
    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle and closest values when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)
        print("Left:", closest_left_angle, ",", closest_left_distance)
        print("Right:", closest_right_angle, ",", closest_right_distance)
        print("Front:", closest_front_angle, ",", closest_front_distance)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
