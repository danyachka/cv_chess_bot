from dataclasses import dataclass
import numpy as np
from cv2.typing import MatLike

from src.cv.contours.square import Square


@dataclass
class Chessboard:
    wrapped: MatLike
    mean_dx: int
    mean_dy: int
    
    def corners_of(self, row, col) -> np.ndarray:
        h, w, _ = self.wrapped.shape

        horizontal_offset = np.round(w / (2 * self.mean_dx * 8))
        vertical_offset = np.round(h / (2 * self.mean_dy * 8))

        x = horizontal_offset + col * self.mean_dx
        y = vertical_offset + row * self.mean_dy
        return np.array([[x, y], [x, y+self.mean_dy], [x+self.mean_dx, y+self.mean_dy], [x+self.mean_dx, y]], dtype=np.int32)


@dataclass
class Grid:
    coords: list[list[float]]

    def print(self):
        for row in self.coords:
            print(*["**" if s is not None else "__" for s in row])


def build_chess_board(rotated_image: MatLike, rotated_squares: list[Square], is_test=False) -> Chessboard:
    grid, mean_w, mean_h = __create_grid(rotated_squares)
    if is_test:
        grid.print()

    if __is_borders_empty(grid):
        print("Empty borders!")
        return None

    min_x = np.mean([row[0].x for row in grid.coords if row[0] is not None])
    min_y = np.mean([s.y for s in grid.coords[0] if s is not None])

    max_x = np.mean([(row[-1].x + row[-1].w) for row in grid.coords if row[-1] is not None])
    max_y = np.mean([(s.y + s.h) for s in grid.coords[-1] if s is not None])

    wrapped = rotated_image[int(min_y):int(max_y), int(min_x):int(max_x)]
    h, w, _ = wrapped.shape
    return Chessboard(
        wrapped=wrapped,
        mean_dx=w//8,
        mean_dy=h//8
    )


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
        print(d_rows, current_row)
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


    print(len(squares))
    for s in squares:
        print(s.row_num, s.col_num)
        grid[s.row_num][s.col_num] = s
    
    return Grid(grid), mean_w, mean_h

    
