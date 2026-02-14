import cv2
import numpy as np


def my_roll_number():
    return "MT2025732"


def count_hough_lines(image,
                      blurring_kernel=(7, 7),
                      morphing_factor=25,
                      canny_low_threshold=0.065,
                      canny_high_threshold=0.195,
                      canny_aperture_size=3,
                      hough_rho=1,
                      hough_theta=np.pi / 180 / 20,
                      hough_threshold=0.104,
                      hough_min_line_length=0.5,
                      hough_max_line_gap=0.104,
                      horizontal_line_tolerance=5,
                      merging_tolerance=0.02,
                      return_intermediate_steps=False,
                      *args, **kwargs):
    # Numpy-ize the input for its rich api
    if not isinstance(image, np.ndarray):  image = np.array(image)
    height, width = image.shape
    intermediate_steps = {'original': image}

    # Apply gray-scaling to simplify computations
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if return_intermediate_steps: intermediate_steps['gray-scaled'] = gray

    # Apply blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, blurring_kernel, 0)
    if return_intermediate_steps: intermediate_steps['blurred'] = blurred

    # Quantize the image to get rid of noise
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    if return_intermediate_steps: intermediate_steps['thresholded'] = thresh

    # Morph the image to reduce non-horizontal artifacts
    kernel_length = width // morphing_factor
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 3))
    morphed = 255 - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    if return_intermediate_steps: intermediate_steps['morphed'] = morphed

    # Apply Canny Edge Detection
    edges = cv2.Canny(morphed, int(width * canny_low_threshold), int(width * canny_high_threshold), apertureSize=canny_aperture_size)
    if return_intermediate_steps: intermediate_steps['edged'] = edges

    # Apply Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=hough_rho,
        theta=hough_theta,
        threshold=int(width * hough_threshold),
        minLineLength=int(width * hough_min_line_length),
        maxLineGap=int(width * hough_max_line_gap),
    )

    if return_intermediate_steps:
        reconstruct = np.ones_like(image) * 255
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(reconstruct, (x1, y1), (x2, y2), 0, 1)
        intermediate_steps['all lines'] = reconstruct

    # Filter in horizontal lines
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < horizontal_line_tolerance:
            horizontal_lines.append((x1, y1, x2, y2))

    # Merge lines which are too close (two thin lines formed as outline of a thicker line in original)
    line_intercepts = [((y1 + y2) / 2, x1, y1, x2, y2) for (x1, y1, x2, y2) in horizontal_lines]
    line_intercepts.sort(key=lambda x: x[0])

    merged_lines = []
    merging_tolerance = int(width * merging_tolerance)  # vertical pixels

    for info in line_intercepts:
        y_avg, x1, y1, x2, y2 = info
        if not merged_lines:
            merged_lines.append(info)
        else:
            last_y = merged_lines[-1][0]

            # If too close, merge
            if abs(y_avg - last_y) < merging_tolerance:
                prev = merged_lines[-1]
                prev_len = abs(prev[3] - prev[1])
                curr_len = abs(x2 - x1)

                # But retain the longer one only
                if curr_len > prev_len:
                    merged_lines[-1] = info
            else:
                merged_lines.append(info)

    if return_intermediate_steps:
        reconstruct = np.ones_like(image) * 255
        for line in merged_lines:
            _, x1, y1, x2, y2 = line
            cv2.line(reconstruct, (x1, y1), (x2, y2), 0, 1)
        intermediate_steps['merged lines'] = reconstruct

    if return_intermediate_steps:
        return len(merged_lines), intermediate_steps

    return len(merged_lines)
