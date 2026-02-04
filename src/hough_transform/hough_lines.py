import cv2
import numpy as np
from numpy.ma.extras import average


def my_roll_number():
    return "MT2025732"


def draw_horizontal_lines(image, lines):
    """
    :param image:
    :param lines: List of tuples of (alpha, distance)
    :return: image
    """
    height, width = image.shape
    for line in lines:
        alpha, distance = line
        alpha = np.deg2rad(alpha)
        c = distance - np.tan(alpha) * (width / 2)
        cv2.line(image,
                 (0, round(c)),
                 (width, round(np.tan(alpha) * width + c)),
                 0, 2)
    return image


def is_point_of_interest(
        value,
        average_energy=None,
        image=None,
        heat_map=None,
):
    if average_energy is not None:
        return value < average_energy
    return value < 0.5


def get_horizontal_lines(
        input_image,
        alpha_variance=5.0, alpha_sample_size=11,
        distance_sample_size_factor=1.0,
        threshold=10,

        return_parameter_space=False
):
    """
    A more specialized version of Hough Transform for Line Detection.
    Instead of looking out for all lines, possibly present in the image,
    we only look for lines which are +-alpha_variance degrees of angle with Horizontal.
    Within that range, we use alpha_sample_size to sample lines in that range.

    Additionally, most angles with short distances are wasted since those lines can not be
    easily drawn in the image. Therefore, I draw the alpha angle and distance from the
    middle of the bottom edge of the image!

    y = mx + c
    The equation must satisfy the slope given by alpha
    -> y = tan(alpha) x + c
    The equation must satisfy (height, width / 2) in image space or (width / 2, height) in geometry
    -> height = tan(alpha) (width / 2) + c
    -> c = height - tan(alpha) (width / 2)
    -> y = tan(alpha) x + height - tan(alpha) (width / 2)
    -> y = (sin(alpha) / cos(alpha)) x + height - (sin(alpha) / cos(alpha)) (width / 2)
    -> cos(alpha) y = sin(alpha) x + cos(alpha) height - sin(alpha) (width / 2)
    -> sin(alpha) (width / 2) - cos(alpha) height = sin(alpha) x - cos(alpha) y
    -> LHS = sin(alpha) (width / 2) - cos(alpha) height
       RHS = sin(alpha) x - cos(alpha) y
    At some arbitrary distance from top,
    -> LHS = sin(alpha) (width / 2) - cos(alpha) distance

    :param input_image:
    :param alpha_variance:
    :param alpha_sample_size:
    :param distance_sample_size_factor:
    :param threshold:
    :return:
    """
    height, width = input_image.shape
    l_alpha = np.linspace(-alpha_variance, alpha_variance, alpha_sample_size)
    l_distance = np.linspace(0, height, int(height * distance_sample_size_factor + 1))

    print(l_alpha, l_alpha.shape)
    print(l_distance, l_distance.shape)

    n_alpha = len(l_alpha)
    n_distance = len(l_distance)

    parameter_space_hist = np.zeros(shape=(n_alpha, n_distance), dtype=np.int32)

    average_energy = input_image.mean()
    for y_h in range(height):
        for x_w in range(width):
            value = input_image[y_h, x_w]
            if not is_point_of_interest(value, average_energy=average_energy): continue
            for i_alpha in range(n_alpha):
                for i_distance in range(n_distance):
                    alpha, distance = np.deg2rad(l_alpha[i_alpha]), l_distance[i_distance]
                    c = distance - np.tan(alpha) * (width / 2)
                    # y = tan(alpha) x + c
                    if np.abs(y_h - np.tan(alpha) * x_w - c) < threshold:
                        parameter_space_hist[i_alpha, i_distance] += 1

    results = {}
    results['lines'] = []
    if return_parameter_space:
        results['parameter_space'] = {'l_alpha': l_alpha, 'l_distance': l_distance}

    return results


def generate_test_sample(
        size=1024,
        location=(0.5, 0.5),
        rotation=0,
        thickness=2,
):
    # Image is in yx plane
    # OpenCV is in xy plane
    image = np.ones((size, size), dtype=np.uint8) * 255
    position = (int(size * location[0]), int(size * location[1]))
    distance = position[1] - (size / 2 - position[0]) * np.tan(np.deg2rad(rotation))
    image = draw_horizontal_line(image, position, rotation, distance, thickness)
    return image


def draw_horizontal_line(image, position, alpha, distance, thickness):
    height, width = image.shape
    alpha = -alpha  # Image coords are in fouth quadrant
    alpha = np.deg2rad(alpha)  # Numpy uses Radian, and my brain favours degrees
    c = distance - np.tan(alpha) * (width / 2)
    cv2.line(image,
             (0, round(c)),
             (width, round(np.tan(alpha) * width + c)),
             0, thickness)
    return image


def calculate_horizontal_line_response(image, pixel, alpha, distance):
    height, width = image.shape
    alpha = np.deg2rad(-alpha)
    c = distance - np.tan(alpha) * (width / 2)
    return np.tan(alpha) * pixel[0] + c - pixel[1]


def count_hough_lines(img, algorithm='HORIZONTAL_FOCUS', **kwargs):
    if algorithm == 'HORIZONTAL_FOCUS':
        return len(get_horizontal_lines(img, **kwargs))
    raise ValueError("Unknown algorithm: {}".format(algorithm))
