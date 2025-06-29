import numpy as np
from cv2.typing import MatLike
from cv2 import getPerspectiveTransform, warpPerspective

from src.cv.chessboard.chessboard import Chessboard
from src.cv.chessboard.grid import Grid, create_grid
from src.cv.chessboard.chessboard_position_check import build_positions
from src.cv.chessboard.grid_expanding import expand_grid
from src.cv.contours.square import Square


def build_chess_board(rotated_image: MatLike, rotated_squares: list[Square], is_white_sided, is_test=False) -> Chessboard:
    grid = create_grid(rotated_squares)
    if is_test:
        grid.print()

    if __is_borders_empty(grid):
        grid = expand_grid(rotated_image, grid, rotated_squares, is_test)

    if __is_borders_empty(grid):
        print("Empty borders!")
        return None
    
    wrapped = __get_wrapped_chessboard(grid, rotated_image, is_white_sided)
    h, w = wrapped.shape[:2]
    dx, dy = w/8, h/8
    return Chessboard(
        wrapped=wrapped,
        mean_dx=dx,
        mean_dy=dy,
        positions=build_positions(dx, dy, wrapped)
    )

def __get_wrapped_chessboard(grid: Grid, rotated_image: MatLike, is_white_sided: bool):
    left, top, right, bottom = [], [], [], []
    
    for i in range(8):
        l_c = grid.coords[i][0]
        if l_c is not None:
            left.append(l_c.approx[0])
            left.append(l_c.approx[1])

        t_c = grid.coords[0][i]
        if t_c is not None:
            top.append(t_c.approx[0])
            top.append(t_c.approx[3])

        r_c = grid.coords[i][-1]
        if r_c is not None:
            right.append(r_c.approx[3])
            right.append(r_c.approx[2])

        b_c = grid.coords[-1][i]
        if b_c is not None:
            bottom.append(b_c.approx[1])
            bottom.append(b_c.approx[2])
    left, top, right, bottom = np.array(left), np.array(top), np.array(right), np.array(bottom)

    left_l, top_l, right_l, bottom_l = __calc_line(left), __calc_line(top), __calc_line(right), __calc_line(bottom)

    points = np.float32([
        __calc_intersection(left_l, top_l),
        __calc_intersection(left_l, bottom_l),
        __calc_intersection(bottom_l, right_l),
        __calc_intersection(right_l, top_l),
    ])

    # print(f"Intersections: {Fore.MAGENTA}{points}{Fore.RESET}")

    h, w = 1200, 1200
    end_points = (
        np.float32([[0, 0], [0, h], [w, h], [w, 0]]) if is_white_sided
        else np.float32([[w, h], [w, 0], [0, 0], [0, h]])
    )
    M = getPerspectiveTransform(points, end_points)

    wrapped = warpPerspective(rotated_image, M, (h, w))
    return wrapped


def __calc_line(points: np.ndarray) -> tuple[float, float]:
    k, b = np.polyfit([p[0] for p in points], [p[1] for p in points], 1)
    return k, b


def __calc_intersection(l1, l2) -> tuple[int, int]:
    k1, b1 = l1
    k2, b2 = l2
    x = abs((b1 - b2) / (k1 - k2))
    y = abs(k1 * x + b1)
    return int(x), int(y)


def __is_borders_empty(grid: Grid) -> bool:
    def is_array_empty(array):
        for el in array:
            if el is not None:
                return False
        return True
    # rows
    if (is_array_empty(grid.coords[0])): 
        return True
    if (is_array_empty(grid.coords[-1])): 
        return True
    # cols
    if (is_array_empty([row[0] for row in grid.coords])): 
        return True
    if (is_array_empty([row[-1] for row in grid.coords])): 
        return True
    return False

    
