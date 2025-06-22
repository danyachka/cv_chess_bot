import cv2
from cv2.typing import MatLike
import numpy as np

from src.cv import utils


def find_chessboard(image: MatLike, is_test=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.blur(gray, (3, 3))
    sobel_image = utils.process_sobel(blurred, 3)

    corners = cv2.cornerHarris(sobel_image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    max_val = corners.max()
    print(max_val)

    if is_test:
        dest_image = np.copy(image)
        dest_image[corners > 0.02 * max_val] = [0, 0, 255]
        utils.show_image(dest_image)
        utils.show_image(corners * 255/max_val)
