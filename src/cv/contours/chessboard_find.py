import cv2
from cv2.typing import MatLike
import numpy as np

from src.cv import utils
from src.cv.contours.square import filter_squares, cluster_squares, Square


def find_chessboard(image: MatLike, is_test=False) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel = utils.process_sobel(gray)
    edges = utils.get_edges(gray=sobel, iterations=0)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = filter_squares(contours)
    print(f"Total squares: {len(squares)}")

    if is_test:
        __add_fake_squares(squares)

    clustered = cluster_squares(squares)

    if is_test:
        __show_test_images(image, edges, clustered)

    return None


def __show_test_images(image, edges, squares: list[list[Square]]):
    utils.show_image(edges)
    # print(f"Found squares: {squares}")
    
    line_img = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    for i in range(len(squares)):
        __draw_squares(line_img, squares[i], colors[i%len(colors)])

    utils.show_image(line_img)

def __add_fake_squares(squares: list[Square]):
    squares.append(Square(20, 20, 40, 40, 1600, np.array([
        [[20, 20]], [[20, 60]], [[60, 60]], [[60, 20]]
    ])))
    squares.append(Square(200, 200, 400, 400, 160000, np.array([
        [[200, 200]], [[200, 600]], [[600, 600]], [[600, 200]]
    ])))
    squares.append(Square(200, 200, 900, 900, 900*900, np.array([
        [[200, 200]], [[200, 200+900]], [[1100, 1100]], [[200+900, 200]]
    ])))

def __draw_squares(line_img: MatLike, squares: list[Square], color) -> None:
    color = list(np.random.random(size=3) * 256)
    for square in squares:
        cv2.drawContours(line_img, [square.approx], 0, color, 2)

