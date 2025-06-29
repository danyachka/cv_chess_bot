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

    new_squares: list[Square] = []
    for x, y, _ in circles:
        if close_x-x_0 <= x <= far_x-x_0 and close_y-y_0 <= y <= far_y-y_0:
            continue
        # tl_x = int(x_0 + mean_w*np.floor(x / mean_w))
        # tl_y = int(y_0 + mean_h*np.floor(y / mean_h))
        tl_x = int(x_0 + x - 0.5*mean_w)
        tl_y = int(y_0 + y - 0.5*mean_h)

        new_squares.append(
            Square(
                tl_x, tl_y, mean_w, mean_h, 
                area=mean_w*mean_h,
                approx=np.int32([
                    (tl_x, tl_y),
                    (tl_x, tl_y + mean_h),
                    (tl_x + mean_w, tl_y + mean_h),
                    (tl_x + mean_w, tl_y),
                ])
            )
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

