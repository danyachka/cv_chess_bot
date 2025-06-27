from dataclasses import dataclass
import numpy as np
from cv2.typing import MatLike
from cv2 import getPerspectiveTransform, warpPerspective
from colorama import Fore

from src.cv.chessboard.chessboard import Chessboard
from src.cv.chessboard.chessboard_position_check import build_positions
from src.cv.contours.square import Square


@dataclass
class Grid:
    coords: list[list[Square]]

    def print(self):
        for row in self.coords:
            print(*["**" if s is not None else "__" for s in row])


def build_chess_board(rotated_image: MatLike, rotated_squares: list[Square], is_white_sided, is_test=False) -> Chessboard:
    grid, _, _ = __create_grid(rotated_squares)
    if is_test:
        grid.print()

    if __is_borders_empty(grid):
        print("Empty borders!")
        return None
    
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

    print(f"Intersections: {Fore.MAGENTA}{points}{Fore.RESET}")

    h, w = 1200, 1200
    end_points = (
        np.float32([[0, 0], [0, h], [w, h], [w, 0]]) if is_white_sided
        else np.float32([[w, h], [w, 0], [0, 0], [0, h]])
    )
    M = getPerspectiveTransform(points, end_points)

    wrapped = warpPerspective(rotated_image, M, (h, w))
    dx, dy = w/8, h/8
    return Chessboard(
        wrapped=wrapped,
        mean_dx=dx,
        mean_dy=dy,
        positions=build_positions(dx, dy, wrapped)
    )


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



def __create_grid(squares: list[Square]) -> tuple[Grid, float, float]:
    if not squares:
        return Grid([[None for _ in range(8)] for _ in range(8)])
    grid = [[None for _ in range(8)] for _ in range(8)]

    mean_w = np.mean([s.w for s in squares])
    mean_h = np.mean([s.h for s in squares])
    
    # rows
    squares_y_sorted = sorted(squares, key=lambda s: s.y)
    current_row = 0
    last_square = squares_y_sorted[0]
    last_square.row_num = current_row

    for i in range(1, len(squares_y_sorted)):
        current_square = squares_y_sorted[i]
        d_rows = round((current_square.y - last_square.y) / mean_w)
        current_row += d_rows
        current_square.row_num = min(current_row, 7)
        last_square = current_square
        
    # cols
    squares_x_sorted = sorted(squares, key=lambda s: s.x)
    current_col = 0
    last_square = squares_x_sorted[0]
    last_square.col_num = current_col

    for i in range(1, len(squares_x_sorted)):
        current_square = squares_x_sorted[i]
        d_rows = round((current_square.x - last_square.x) / mean_h)
        current_col += d_rows
        current_square.col_num = min(current_col, 7)
        last_square = current_square


    for s in squares:
        grid[s.row_num][s.col_num] = s
    
    return Grid(grid), mean_w, mean_h

    
