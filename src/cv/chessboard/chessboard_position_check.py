from cv2.typing import MatLike
import numpy as np
import cv2

from src.cv.chessboard.chessboard import corners_of, Position
from src.cv import utils


def build_positions(dx: float, dy: float, wrapped: MatLike) -> list[list[Position]]:
    result = []
    for row in range(8):
        row_array = []
        for col in range(8):
            corners = corners_of(dx, dy, row, col)
            row_array.append(
                define_position_type(
                    wrapped[
                        corners[0][1] : corners[2][1], corners[0][0] : corners[2][0]
                    ],
                    is_black_cell=(8*row + col) % 2 == 0,
                    # is_test=(row==7 and col==7)
                )
            )
        result.append(tuple(row_array))
    return tuple(result)


def define_position_type(cell_image: np.ndarray, is_black_cell: bool, is_test=False) -> Position:
    cell_size = cell_image.shape[0]
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blurred, 60, 120, None, 3)

    _, edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    thresh = edges

    contours = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, 
        minDist=1,
        param1=255, 
        param2=18,
        minRadius=int(cell_size/5), 
        maxRadius=int(cell_size/1.8)
    )
    if is_test:
        utils.show_image(cell_image)
        utils.show_image(thresh)

    if contours is None:
        return Position.EMPTY
    circles = np.uint16(contours[0, :])

    if is_test:
        __show_drafted_circles(circles, cell_image)
    
    center = (cell_size // 2, cell_size // 2)
    max_area = 0.85 * (cell_size ** 2)
    
    valid_contours = []
    for circle in circles:
        area = np.pi * circle[2]**2
        
        if area > max_area:
            continue
        
        dist = utils.calc_points_dist(center, (circle[0], circle[1]))
        if dist > circle[2] * 1.3:
            continue

        valid_contours.append(circle)

    
    if is_test:
        __show_drafted_circles(valid_contours, cell_image)
    
    if len(valid_contours) == 0:
        return Position.EMPTY
    
    main_contour = max(valid_contours, key=lambda el: el[2])
    
    mask = np.zeros_like(gray)
    cv2.circle(mask, (main_contour[0], main_contour[1]), main_contour[2], 255, -1)
    
    if is_test:
        utils.show_image(mask)
    
    mean_color_bgr = cv2.mean(cell_image, mask=mask)[:3]
    mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    if mean_color_hsv[2] > 120: 
        piece_color = Position.WHITE
    elif mean_color_hsv[2] < 80:
        piece_color = Position.BLACK
    else:
        if is_black_cell:
            piece_color = Position.BLACK if mean_color_hsv[2] < 160 else Position.WHITE
        else:
            piece_color = Position.WHITE if mean_color_hsv[2] > 100 else Position.BLACK
    
    return piece_color


def __show_drafted_circles(circles: np.ndarray, image) -> None:
    print("Contours:", len(circles))
    img = image.copy()
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    utils.show_image(img)