import cv2
import numpy as np

from hough_transform import *
from itertools import product


def test_playground_copied_for_convenience():
    roll = "MT2025732"
    name = "Anirudh Sharma"

    img1 = blank_ruled_paper(roll)
    img2 = printed_paper(roll, name, roll)
    img3 = handwritten_paper(roll, name, roll)

    # Code to visualize the generated image.
    cv2.imshow("image", img1)
    cv2.waitKey(0)
    cv2.imshow("image", img2)
    cv2.waitKey(0)
    cv2.imshow("image", img3)
    cv2.waitKey(0)

    print("Blank ruled lines:", count_hough_lines(img1))
    print("Printed lines:", count_hough_lines(img2))
    print("Handwritten lines:", count_hough_lines(img3))


def test_generate_test_sample():
    image = generate_test_sample()
    assert image[512, 512] == 0
    assert image[512, 0] == 0
    image = generate_test_sample(rotation=45)
    assert image[512, 512] == 0
    assert image[-1, 0] == 0
    image = generate_test_sample(location=(0.25, 0.25))
    assert image[256, 512] == 0
    assert image[256, 0] == 0
    image = generate_test_sample(location=(0.25, 0.25), rotation=-45)
    assert image[256, 256] == 0
    assert image[0, 0] == 0


def test_calculate_horizontal_line_response():
    image = generate_test_sample(location=(0.5, 0.25))
    response = calculate_horizontal_line_response(image, (12, 256), 0, 256)
    assert response == 0.0
    response = calculate_horizontal_line_response(image, (12, 255), 0, 256)
    assert response == 1.0
    response = calculate_horizontal_line_response(image, (12, 258), 0, 256)
    assert response == -2.0
    response = calculate_horizontal_line_response(image, (512, 512), 45, 512)
    assert response == 0.0
    response = calculate_horizontal_line_response(image, (511, 511), 45, 512)
    assert response == 2.0
    response = calculate_horizontal_line_response(image, (514, 514), 45, 512)
    assert np.isclose(response, -4.0)


def test_blank_ruled_paper():
    image, truth = blank_ruled_paper("test_blank_ruled_paper", H=300, W=200, return_truth=True)
    print()

    cv2.imshow("image", image)
    cv2.waitKey(0)

    # prediction = count_hough_lines(image)
    result = get_horizontal_lines(image, distance_sample_size_factor=0.5, return_parameter_space=True)
    parameter_space = list(product(result['parameter_space']['l_alpha'], result['parameter_space']['l_distance']))
    lined_image = draw_horizontal_lines(image, parameter_space)
    cv2.imshow("lined_image", lined_image)
    cv2.waitKey(0)
    # assert truth == prediction
