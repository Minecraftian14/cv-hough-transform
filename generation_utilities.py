import cv2
import random as rd
import numpy as np

with open("lorem_ipsum.txt", "r") as f:
    lorem_ipsum = np.asarray(f.read().split(" "))

default_parameters = {
    'p_margin': (0.1, 0.1, 0.1, 0.15),
    'p_texture_noise': (0, 10),
    'p_line_spacing': (60, 4),
    'p_line_angle': (1, 0.5),
    'p_line_offset': (0, 5),
    'p_line_thickness': (0.003, 0.0003),
    'p_line_intensity': (0, 40),
    'p_line_dropout': 0.1,
    'p_line_breaker': (2, 30, 10),
    'p_vignette_blur': 2,
}


def lerp1D(a, b, f):
    return int(a + (b - a) * f)


def lerp2D(a, b, f):
    return lerp1D(a[0], b[0], f), lerp1D(a[1], b[1], f)


def generate_paper(
        # Base Parameters
        seed: int = None,
        text: str = None,
        height: int = 1024,
        width: int = 768,
        # Random Engine Parameters
        p_margin: tuple[float, float, float, float] = default_parameters['p_margin'],
        p_texture_noise: tuple[int, int] = default_parameters['p_texture_noise'],
        p_line_spacing: tuple[int, int] = default_parameters['p_line_spacing'],
        p_line_angle: tuple[float, float] = default_parameters['p_line_angle'],
        p_line_offset: tuple[int, int] = default_parameters['p_line_offset'],
        p_line_thickness: tuple[float, float] = default_parameters['p_line_thickness'],
        p_line_intensity: tuple[int, int] = default_parameters['p_line_intensity'],
        p_line_dropout: float = default_parameters['p_line_dropout'],
        p_line_breaker: tuple[float, int, int] = default_parameters['p_line_breaker'],
        p_vignette_blur: int = default_parameters['p_vignette_blur'],
        # Control Parameters
        debug: bool = False,
):
    """
    :param seed: Randomizer seed.
    :param text: Text to be written on paper.
    :param height: Height of paper.
    :param width: Width of paper.
    :param p_margin: Relative margin around paper. (E, N, W, S)
    :param p_texture_noise: (mean, std) pair for added gaussian noise.
    :param p_line_spacing: (mean, std) pair for line-spacing.
    :param p_line_angle: (mean, std) pair for line-angle from horizontal.
    :param p_line_offset: (mean, std) pair for line-offset from margin.
    :param p_line_thickness: (mean, std) pair for line-thickness from margin.
    :param p_line_intensity: (mean, std) pair for line-intensity from margin.
    :param p_line_dropout: Probability that a line is not even drawn.
    :param p_line_breaker: (existence std, size mean, size std) triplet for drawing broken lines.
    :param p_vignette_blur: Radial blur centered to page.
    :param debug: Debug mode draws.
    :return: The final generated image and the ground truth pair.
    """

    if seed is not None:
        rd.seed(seed)
        np.random.seed(seed)

    if text is None:
        words = lorem_ipsum.copy()
        rd.shuffle(words)
    else:
        words = text.split(" ")

    h, w, h2, w2 = height, width, height // 2, width // 2
    line_thickness = max(1, int(p_line_thickness[0] * width))
    if debug:  print("Line Thickness:", line_thickness)
    image = np.ones((h, w), dtype=np.float16) * 255

    margin = (int(w * p_margin[0]), int(h * p_margin[1]), int(w * (1 - p_margin[2])), int(h * (1 - p_margin[3])))
    if debug:  cv2.rectangle(image, margin[:2], margin[2:], 230, line_thickness)

    texture_noise = np.random.normal(p_texture_noise[0], p_texture_noise[1], (h, w))
    image = image + texture_noise
    if debug:  cv2.line(image, (w2, h2), (w2 + p_texture_noise[1] * 2, h2), 0, line_thickness)

    word_index = 0
    line_height = margin[1] - p_line_spacing[0]  # -1 indexing
    line_width = margin[2] - margin[0]
    line_count = 0

    font, font_thickness = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, int(line_thickness)
    (_, sentence_height), sentence_baseline = cv2.getTextSize("Ag", font, 1.0, font_thickness)
    actual_height = sentence_height + sentence_baseline
    target_height = p_line_spacing[0] * 0.5
    font_scale = target_height / actual_height

    while line_height < margin[3]:
        line_height += p_line_spacing[0]
        if rd.random() < p_line_dropout: continue
        line_count += 1
        line_spacing = int(rd.gauss(0, p_line_spacing[1]))
        line_angle = np.deg2rad(rd.gauss(*p_line_angle))
        line_offset = int(rd.gauss(*p_line_offset))
        line_thickness = max(1, int(rd.gauss(*p_line_thickness) * width))
        line_intensity = abs(int(rd.gauss(*p_line_intensity)))
        n_breakages = int(abs(rd.gauss(0, p_line_breaker[0])))

        line_height_opposite = int((w2 - margin[0]) * np.tan(line_angle))
        pt1 = (margin[0] + line_offset, line_height + line_spacing - line_height_opposite)
        pt2 = (margin[2] + line_offset, line_height + line_spacing + line_height_opposite)

        if n_breakages == 0:
            breakages = []
        else:
            # Get breakage position as a fraction of the full line length
            breakages = [line_width * rd.random() for _ in range(n_breakages)]
            # Also get the breakage width
            breakages = [(position, rd.gauss(*p_line_breaker[1:])) for position in breakages]
            # Convert them to start and stop positions of the breakage, and sort for internal merge
            breakages = sorted([[position - breakage_width / 2, position + breakage_width / 2]
                                for position, breakage_width in breakages])
            # Merge overlapping breakages
            _breakages = [breakages[0]]
            for breakage in breakages[1:]:
                if breakage[0] <= _breakages[-1][1]:
                    _breakages[-1][1] = breakage[1]
                else:
                    _breakages.append(breakage)
            breakages = _breakages
            # Convert them to normal factors of line length for interpolation
            breakages = [(start / line_width, stop / line_width) for start, stop in breakages]

        start = 0
        for a, b in breakages:
            cv2.line(image, lerp2D(pt1, pt2, start), lerp2D(pt1, pt2, a),
                     line_intensity, line_thickness)
            start = b
        cv2.line(image, lerp2D(pt1, pt2, start), lerp2D(pt1, pt2, 1.0),
                 line_intensity, line_thickness)

        if text == '': continue
        sentence = ""
        while True:
            new_sentence = sentence + (" " if sentence else "") + words[word_index]
            (sentence_width, _), _ = cv2.getTextSize(new_sentence, font, font_scale, font_thickness)
            if sentence_width > line_width: break
            sentence = new_sentence
            word_index += 1
            if word_index >= len(words): word_index = 0
        (sentence_width, sentence_height), sentence_baseline = cv2.getTextSize(sentence, font, font_scale, font_thickness)
        text_image = np.ones((sentence_height + sentence_baseline + 10, sentence_width + 10)) * 255
        cv2.putText(text_image, sentence, (5, sentence_height + 5), font, font_scale, 0, font_thickness, cv2.LINE_AA)
        text_image_center = (text_image.shape[1] // 2, text_image.shape[0] // 2)
        rotation_affine = cv2.getRotationMatrix2D(text_image_center, np.rad2deg(-line_angle), 1.0)
        text_image = cv2.warpAffine(text_image, rotation_affine, text_image.shape[::-1],
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        context = image[pt1[1] - text_image.shape[0] + sentence_baseline:pt1[1] + sentence_baseline, pt1[0]:pt1[0] + text_image.shape[1]]
        mask = text_image < 128
        context[mask] = text_image[mask]

    if p_vignette_blur > 0:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=p_vignette_blur)
        y, x = np.indices((h, w))
        dists = np.sqrt((y - h2) ** 2 + (x - w2) ** 2)
        mask = dists / dists.max()
        mask = mask ** 1.5  # make it smoother
        image = image * (1 - mask) + blur * mask

    return image.clip(0, 255).astype(np.uint8), line_count


if __name__ == '__main__':
    print("")
    paper, count = generate_paper(debug=False,
                                  p_margin=(0.1, 0.1, 0.1, 0.15),
                                  p_texture_noise=(0, 10),
                                  p_line_spacing=(60, 4),
                                  p_line_angle=(1, 0.5),
                                  p_line_offset=(0, 5),
                                  p_line_thickness=(0.003, 0.0003),
                                  p_line_intensity=(0, 40),
                                  p_line_dropout=0.1,
                                  p_line_breaker=(2, 30, 10),
                                  p_vignette_blur=2,
                                  )
    print(count)
    import matplotlib.pyplot as plt

    plt.imshow(paper, cmap="gray_r", vmin=0, vmax=255)
    plt.show()
