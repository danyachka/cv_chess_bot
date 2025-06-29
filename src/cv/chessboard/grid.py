from dataclasses import dataclass

import numpy as np

from src.cv.contours.square import Square


@dataclass
class Grid:
    coords: list[list[Square]]

    def print(self):
        for row in self.coords:
            print(*["**" if s is not None else "__" for s in row])

    def calc_empty_stats(self) -> tuple[float, float, float, float]:
        side_empty_rows_count = 0
        side_empty_cols_count = 0

        most_far_x = 0
        most_far_y = 0

        for i in range(0, 8):
            row = [s for s in self.coords[7 - i] if s is not None]
            if len(row) == 0:
                side_empty_rows_count += 1
            else:
                most_far_y = max([s.approx[2, 1] for s in row])
                break

        for i in range(0, 8):
            col = [row[7 - i] for row in self.coords if row[7 - i] is not None]
            if len(col) == 0:
                side_empty_cols_count += 1
            else:
                most_far_x = max([s.approx[2, 0] for s in col])
                break

        return most_far_x, side_empty_cols_count, most_far_y, side_empty_rows_count
    
    def get_closest_coords(self) -> tuple[int, int]:
        x = min([row[0].approx[0, 0] for row in self.coords if row[0] is not None])
        y = min([s.approx[0, 1] for s in self.coords[0] if s is not None])
        return x, y
    

def create_grid(squares: list[Square]) -> Grid:
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
    
    return Grid(grid)
