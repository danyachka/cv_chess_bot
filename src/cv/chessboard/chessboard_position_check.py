from cv2.typing import MatLike
import numpy as np
import cv2

from src.cv.chessboard.chessboard import corners_of, Position
from src.cv import utils


def build_positions(dx: float, dy: float, wrapped: MatLike) -> list[list[Position]]:
    result = []
    for i in range(8):
        row = []
        for j in range(8):
            corners = corners_of(dx, dy, i, j)
            row.append(
                define_position_type(
                    wrapped[
                        corners[0][0] : corners[2][0], corners[0][1] : corners[2][1]
                    ],
                    is_black_cell=(8*i + j) % 2 == 0
                )
            )
        result.append(tuple(row))
    return tuple(result)


def define_position_type(cell_image: np.ndarray, is_black_cell: bool, is_test=False) -> Position:
    cell_size = cell_image.shape[0]
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    thresh = utils.get_edges(gray, 0)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = cv2.HoughCircles(
        thresh, cv2.HOUGH_GRADIENT, 1, 
        minDist=2,
        param1=255, 
        param2=13,
        minRadius=int(cell_size*0.33), 
        maxRadius=int(cell_size*0.9)
    )
    if contours is None:
        return Position.EMPTY
    circles = np.uint16(contours[0, :])

    if is_test:
        utils.show_image(thresh)
        __show_drafted_circles(circles, cell_image)
    
    if len(circles) == 0:
        return Position.EMPTY
    
    center = (cell_size // 2, cell_size // 2)
    min_area = 0.2 * (cell_size ** 2)
    max_area = 0.85 * (cell_size ** 2)
    
    valid_contours = []
    for circle in circles:
        area = np.pi * circle[2]**2
        
        if area < min_area or area > max_area:
            continue
        
        dist = utils.calc_points_dist(center, (circle[0], circle[1])) * 1.2
        if dist > circle[2]:
            continue

        valid_contours.append(circle)

    
    if is_test:
        __show_drafted_circles(valid_contours, cell_image)
    
    if len(valid_contours) == 0:
        return Position.EMPTY
    
    main_contour = max(valid_contours, key=lambda el: el[2])
    
    mask = np.zeros_like(gray)
    cv2.circle(mask, (main_contour[0], main_contour[1]), main_contour[2], (0, 255, 0), 2)
    
    mean_color_bgr = cv2.mean(cell_image, mask=mask)[:3]
    mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    if mean_color_hsv[2] > 140: 
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