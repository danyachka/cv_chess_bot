from dataclasses import dataclass
import numpy as np

from src.cv.contours.square import Square


@dataclass
class Grid:
    coords: list[list[float]]

    def print(self):
        total = 0
        for row in self.coords:
            print(*["**" if s is not None else "__" for s in row])
        for i, row in enumerate(self.coords):
            count = len([el for el in row if el is not None])
            total += count
            print(f'{i}: {count}')
        print("Total:", total)



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
        current_square.row_num = current_row
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
        current_square.col_num = current_col
        last_square = current_square
        print(d_rows, current_row)


    print(len(squares))
    for s in squares:
        grid[s.row_num][s.col_num] = s
    
    return Grid(grid)

    
