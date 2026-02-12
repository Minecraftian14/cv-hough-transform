import cv2
import numpy as np
import random

import numpy as np
import hashlib

def seed_from_string(input_string):
    # Use hashlib to create a consistent, large integer hash of the string
    hash_value = int(hashlib.sha256(input_string.encode('utf-8')).hexdigest(), 16) % 1000000
    return hash_value
    
def set_seed(seed_string):
    seed = seed_from_string(seed_string)
    random.seed(seed)
    np.random.seed(seed)

def blank_ruled_paper(seed, H=1024, W=768, return_truth=False):
    set_seed(seed)
    img = np.ones((H, W), dtype=np.uint8) * 255
    spacing = random.randint(30, 50)

    for y in range(50, H, spacing):
        cv2.line(img, (50, y), (W - 50, y), 0, 2)

    if return_truth:
        return img, len(range(50, H, spacing))

    return img

def printed_paper(seed, name, roll, H=1024, W=768):
    set_seed(seed)
    img = blank_ruled_paper(seed, H, W)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Name: {name}", (60, 80),
                font, 1, 0, 2)
    cv2.putText(img, f"Roll: {roll}", (60, 120),
                font, 1, 0, 2)

    return img


import cv2
import numpy as np
import random

def add_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img, prob=0.01):
    out = img.copy()
    mask = np.random.rand(*img.shape)
    out[mask < prob] = 0
    out[mask > 1 - prob] = 255
    return out

def random_line_dropout(img, drop_prob=0.15):
    h, w = img.shape
    for y in range(100, h, 45):
        if random.random() < drop_prob:
            cv2.line(img, (50, y), (w - 50, y), 255, 6)
    return img

def small_rotation(img, max_angle=2.0):
    h, w = img.shape
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

def generate_noisy_handwritten(seed, base_generator):
    random.seed(seed)
    np.random.seed(seed)

    img = base_generator(seed, "Hidden", seed)

    img = random_line_dropout(img)
    img = add_gaussian_noise(img, sigma=12)
    img = add_salt_pepper(img, prob=0.008)
    img = small_rotation(img)

    return img

    
def handwritten_paper(seed, name, roll, H=1024, W=768, return_gt=False):
    #more text will added for hidden test,
    #so please try adding text to moddle of rows.
    set_seed(seed)
    img = np.ones((H, W), dtype=np.uint8) * 255
    
    spacing = random.randint(35, 55)
    y_positions = []

    for y in range(120, H, spacing):
        jitter = random.randint(-5, 5)
        cv2.line(img,
                 (60 + jitter, y),
                 (W - 60 + jitter, y),
                 0,
                 random.randint(1, 3))
        y_positions.append(y)

    cv2.putText(img, f"{name}", (80, 80),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1.2, 0, 2)
    cv2.putText(img, f"{roll}", (80, 120),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1.2, 0, 2)

    #Hint: For hidden testng
    #img = random_line_dropout(img)
    #img = add_gaussian_noise(img, sigma=12)
    #img = add_salt_pepper(img, prob=0.008)
    #img = small_rotation(img)
    
    if return_gt:
        return img, len(y_positions)

    return img



