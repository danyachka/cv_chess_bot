import cv2
from cv2.typing import MatLike
import numpy as np

from src.cv import utils
from src.cv.chessboard.grid import Grid, create_grid
from src.cv.contours.square import Square


def expand_grid(rotated_image: MatLike, grid: Grid, rotated_squares: list[Square], is_test=False) -> Grid:
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

    mean_w = np.mean([s.w for s in rotated_squares])
    mean_h = np.mean([s.h for s in rotated_squares])
    far_x, e_cols, far_y, e_rows = grid.calc_empty_stats()
    close_x, close_y = grid.get_closest_coords()

    m = 1
    x_0 = int(max(0, close_x - e_cols*mean_w*m))
    y_0 = int(max(0, close_y - e_rows*mean_h*m))

    h, w = gray.shape
    x_1 = int(min(w, far_x + e_cols*mean_w*m))
    y_1 = int(min(h, far_y + e_rows*mean_h*m))
    if is_test:
        print(f'{x_0}:{x_1}, {y_0}:{y_1}')
    wrapped = gray[y_0:y_1, x_0:x_1]

    edges = utils.get_edges(gray=wrapped, iterations=1)

    cell_size = (mean_h + mean_w)/2 * m

    contours = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, 1, 
        minDist=0.5 * cell_size,
        param1=255, 
        param2=20,
        minRadius=int(0.7*cell_size/2), 
        maxRadius=int(cell_size/2)
    )

    if contours is None:
        print("No circles")
        return grid
    
    circles = np.uint16(contours[0, :])
    if is_test:
        print(f"Circles detected = {len(circles)}")

    new_squares = __get_new_squares(
        circles, grid, close_x, close_y, far_x, far_y, x_0, y_0, mean_w, mean_h, e_rows, e_cols
    )
    
    if is_test:
        utils.show_image(wrapped)
        image = rotated_image.copy()
        print(len(new_squares))
        
        for s in new_squares:
            cv2.drawContours(image, [s.approx], 0, (255, 0, 255), 2)
        for x, y, r in circles:
            cv2.circle(image, (x_0+x, y_0+y), r, (0, 0, 255), 2)
        utils.show_image(image, "By circles positions")

    rotated_squares.extend(new_squares)

    return create_grid(rotated_squares)


def __get_new_squares(
        circles: list[MatLike],
        grid: Grid,
        close_x, close_y, far_x, far_y,
        x_0, y_0,
        mean_w, mean_h,
        e_rows, e_cols
) -> list[Square]:
    new_squares: list[Square] = []
    for x, y, _ in circles:
        if close_x-x_0 <= x <= far_x-x_0 and close_y-y_0 <= y <= far_y-y_0:
            continue
        # top left corner
        tl_x = int(x_0 + x - 0.5*mean_w)
        tl_y = int(y_0 + y - 0.5*mean_h)

        mean_approx = np.zeros((4, 2), dtype=np.float32)
        counter = 0
        if x < close_x-x_0 or x > far_x-x_0:
            col_n = 0 if x < close_x-x_0 else 7 - e_cols
            for i in range(8):
                s = grid.coords[i][col_n]
                if s is None:
                    continue
                counter += 1
                mean_approx += s.approx - s.approx[0]
        elif y < close_y-y_0 or y > far_y-y_0:
            row_n = 0 if y < close_y-y_0 else 7 - e_rows
            for s in grid.coords[row_n]:
                if s is None:
                    continue
                counter += 1
                mean_approx += s.approx - s.approx[0]
        
        mean_approx = (mean_approx / counter) + (tl_x, tl_y)

        new_squares.append(
            Square(
                tl_x, tl_y, mean_w, mean_h, 
                area=mean_w*mean_h,
                approx=np.int32(mean_approx)
            )
        )
    return new_squares
