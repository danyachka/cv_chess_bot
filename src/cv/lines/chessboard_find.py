import cv2
import numpy as np
from cv2.typing import MatLike
from src.cv import utils
from src.cv.lines.line import Line, LineGroup, group_lines


def find_chessboard(image: MatLike, is_test=False) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = utils.get_edges(gray=gray, iterations=0)

    detected = __detect_lines(edges)
    shape = image.shape
    print(f"Image shape: {shape}")
    lines_groups = group_lines(detected, shape)

    if is_test:
        __show_test_images(image, edges, detected, lines_groups)


def __detect_lines(edges, threshold=100) -> list[Line]:
    lines = cv2.HoughLines(
        edges, rho=1, theta=np.pi / 180, threshold=threshold, lines=None, srn=0, stn=0
    )
    return [Line(line) for line in lines]


def __show_test_images(
    image: MatLike, edges: MatLike, lines: list[Line], lines_groups: list[LineGroup]
):
    utils.show_image(edges)

    image_with_lines = np.copy(image)
    draw_lines(image_with_lines, (0, 255, 0), lines)
    utils.show_image(image_with_lines, "Detected Lines")


    image_with_lines = np.copy(image)
    for group in lines_groups:
        color = list(np.random.random(size=3) * 256)
        draw_lines(image_with_lines, color, group.lines)

    for group in lines_groups:
        for inter in group.intersections:
            cv2.circle(image_with_lines, inter, radius=3, color=(0, 0, 255), thickness=3)
    utils.show_image(image_with_lines, "Filtered Lines")


def draw_lines(
      image: MatLike, color: tuple[int, int, int], lines: list[Line]
):
   for line in lines:
      a = np.cos(line.theta)
      b = np.sin(line.theta)

      x0 = a*line.r
      y0 = b*line.r

      line_len = 3000
      x1 = int(x0 + line_len*(-b))
      y1 = int(y0 + line_len*(a))
      x2 = int(x0 - line_len*(-b))
      y2 = int(y0 - line_len*(a))

      cv2.line(image, (x1, y1), (x2, y2), color, 2)
   return image
