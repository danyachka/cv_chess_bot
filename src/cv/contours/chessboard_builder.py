from dataclasses import dataclass

from src.cv.contours.square import Square


@dataclass
class Grid:
    coords: list[list[float]]

    def print(self):
        for row in self.coords:
            print(*["**" if s is not None else "__" for s in row])



def create_grid(squares: list[Square]) -> Grid:
    if not squares:
        return Grid([[None for _ in range(8)] for _ in range(8)])
    
    squares_sorted_y = sorted(squares, key=lambda s: s.y)
    
    rows = []
    current_row = [squares_sorted_y[0]]
    avg_height = squares_sorted_y[0].h
    
    for square in squares_sorted_y[1:]:
        if abs(square.y - current_row[-1].y) < avg_height * 0.5:  # Similar y
            current_row.append(square)
        else:
            current_row.sort(key=lambda s: s.x)
            rows.append(current_row)
            current_row = [square]
            avg_height = (avg_height * (len(rows) - 1) + square.h) / len(rows)
    
    if current_row:
        current_row.sort(key=lambda s: s.x)
        rows.append(current_row)
    
    grid = [[None for _ in range(8)] for _ in range(8)]
    
    # For columns, we need to look at all rows
    all_cols = []
    for row in rows:
        all_cols.extend([s.x for s in row])
    all_cols = sorted(list(set(all_cols)))  # Get unique x positions
    
    # Now assign to grid cells
    for row_idx, row in enumerate(rows):
        for square in row:
            # Find closest column
            col_idx = min(range(len(all_cols)), key=lambda i: abs(all_cols[i] - square.x))
            # Map to 8x8 grid (assuming we have exactly 8 rows and 8 cols)
            mapped_row = min(7, int(8 * row_idx / len(rows)))
            mapped_col = min(7, int(8 * col_idx / len(all_cols)))
            grid[mapped_row][mapped_col] = square
    
    return Grid(grid)

    
