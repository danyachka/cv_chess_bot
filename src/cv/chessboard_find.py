import time
import cv2
from cv2.typing import MatLike
import numpy as np
from colorama import Fore

from src.cv import utils
from src.cv.chessboard.chessboard import Position
from src.cv.contours.rotation import process_rotation
from src.cv.contours.square import filter_squares, cluster_squares, Square
from src.cv.chessboard.chessboard_builder import build_chess_board, Chessboard


def find_chessboard(image: MatLike, is_white_sided, is_test=False) -> Chessboard:
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sobel = utils.process_sobel(gray)
    edges = utils.get_edges(gray=gray, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = filter_squares(contours)
    if is_test:
        __add_fake_squares(squares)
    clustered = cluster_squares(squares)
    if is_test:
        print(f"Total squares: {len(squares)}")
        print("squares:", len(clustered[0]))

    rotated_image, rotated_squares = process_rotation(image, clustered[0])

    chessboard = build_chess_board(rotated_image, rotated_squares, is_white_sided, is_test=is_test)
    if is_test:
        print(f"{Fore.CYAN}Elapsed time: {time.time() - start}{Fore.RESET}")
        __show_test_images(image, edges, clustered, rotated_image, rotated_squares, chessboard)

    return chessboard


def __show_test_images(
    image,
    edges,
    squares: list[list[Square]],
    rotated_image: MatLike,
    rotated_squares: list[Square],
    chessboard: Chessboard
) -> None:
    utils.show_image(edges)
    # print(f"Found squares: {squares}")
    
    line_img = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    for i in range(len(squares)):
        __draw_squares(line_img, squares[i], colors[i%len(colors)])
    # utils.show_image(line_img)

    line_rotated_image = rotated_image.copy()
    __draw_squares(line_rotated_image, rotated_squares, colors[0])
    utils.show_image(line_rotated_image)
    
    if chessboard is not None:
        wrapped = chessboard.wrapped.copy()
        h, w , _= wrapped.shape
        for i in range(0, 7):
            p = chessboard.corners_of(i, i)[-1]
            cv2.line(wrapped, (0, p[1]), (w, p[1]), (0, 0, 255), 3)
            cv2.line(wrapped, (p[0], 0), (p[0], h), (0, 0, 255), 3)

        for i in range(8):
            for j in range(8):
                p = chessboard.corners_of(i, j)[1]
                position = chessboard.positions[i][j]
                if position == Position.EMPTY:
                    continue
                s = f'{position.name[0]}, ({i}, {j})'
                cv2.putText(wrapped, s, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, 2)

        utils.show_image(wrapped)


def __add_fake_squares(squares: list[Square]):
    squares.append(Square(20, 20, 40, 40, 1600, np.array([
        [20, 20], [20, 60], [60, 60], [60, 20]
    ])))
    squares.append(Square(200, 200, 400, 400, 160000, np.array([
        [200, 200], [200, 600], [600, 600], [600, 200]
    ])))
    squares.append(Square(200, 200, 900, 900, 900*900, np.array([
        [200, 200], [200, 200+900], [1100, 1100], [200+900, 200]
    ])))

def __draw_squares(line_img: MatLike, squares: list[Square], color) -> None:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color = list(np.random.random(size=3) * 256)
    for square in squares:
        cv2.drawContours(line_img, [square.approx], 0, color, 2)
        cv2.circle(line_img, (square.x, square.y), radius=5, color=(255, 255, 0), thickness=5)
        for i, point in enumerate(square.approx[:-1]):
            cv2.circle(line_img, point, radius=2, color=colors[i], thickness=3)

