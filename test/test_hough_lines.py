import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    assert response == -2
    response = calculate_horizontal_line_response(image, (512, 512), 45, 512)
    assert response == 0.0
    response = calculate_horizontal_line_response(image, (511, 511), 45, 512)
    assert np.isclose(response, 2 ** 0.5)
    response = calculate_horizontal_line_response(image, (514, 514), 45, 512)
    assert np.isclose(response, - 8 ** 0.5)


def test_draw_horizontal_lines():
    image = blank_ruled_paper("test_draw_horizontal_lines", H=300, W=200)
    parameter_space_basis = generate_parameter_space_basis(300, 5, 3, 0.05)
    parameter_space = list(product(parameter_space_basis['l_alpha'], parameter_space_basis['l_distance']))
    parameter_space = [(alpha, distance, 1) for alpha, distance in parameter_space]
    image = draw_horizontal_lines(image, parameter_space)
    # Should see 16 triplets
    cv2.imshow("lined_image", image)
    cv2.waitKey(0)


def test_blank_ruled_paper():
    # image, truth = blank_ruled_paper("test_blank_ruled_paper", H=300, W=200, return_truth=True)
    image = handwritten_paper("test_blank_ruled_paper", "Anirudh Sharma", "MT2025732", H=600, W=600)
    print()

    result = get_horizontal_lines(image, -6, 11, 1 / 5, return_parameter_space=True, return_parameter_space_hist=True, threshold=5)
    l_alpha, l_distance = result['parameter_space']['l_alpha'], result['parameter_space']['l_distance']
    histogram = np.asarray(result['parameter_space_hist'])

    print(histogram.shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[1].set_title("Parameter Space Histogram")
    axs[1].imshow(histogram, aspect='auto')
    axs[1].set_ylabel("Alpha")
    axs[1].set_yticks(range(len(l_alpha)), map(lambda x: "%.1f" % x, l_alpha))
    axs[1].set_xlabel("Distance")
    # axs[1].set_xticks(range(len(l_distance)), map(str, l_distance))
    axs[1].legend()
    # pcm = axs[1].pcolormesh(histogram, cmap='viridis')
    # fig.colorbar(pcm, ax=axs[1], shrink=0.6)  # wtf?
    plt.show()

    # assert truth == prediction
