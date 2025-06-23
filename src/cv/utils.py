import cv2
import numpy as np
from cv2.typing import MatLike


def get_edges(gray: MatLike, iterations: int):
    edges = gray

    _, edges = cv2.threshold(edges, 110, 255, cv2.THRESH_BINARY)

    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.Canny(edges, 60, 140, None, 3)

    if iterations != 0:
        edges = increase_lines_thickness(edges=edges, iterations=iterations)

    return edges


def process_sobel(src_image, kernel_size=3):
    grad_x = cv2.Sobel(
        src_image,
        cv2.CV_16S,
        1,
        0,
        ksize=kernel_size,
        scale=1,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )
    grad_y = cv2.Sobel(
        src_image,
        cv2.CV_16S,
        0,
        1,
        ksize=kernel_size,
        scale=1,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def increase_lines_thickness(edges: MatLike, iterations: int):
    kernel = np.ones((3, 3), np.uint8)
    thickened = cv2.dilate(edges, kernel, iterations=iterations)

    return cv2.morphologyEx(thickened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def calc_points_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def calc_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.arctan((y2 - y1) / (x2 - x1))


def show_image(image, tag="Image"):
    cv2.namedWindow(tag, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(tag, image)
    cv2.resizeWindow('custom window', 1000, 1000)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()
