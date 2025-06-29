import cv2
from cv2.typing import MatLike
import numpy as np

from src.cv.contours.square import Square


def process_rotation(image: MatLike, squares: list[Square]) -> tuple[MatLike, list[Square]]:
    horizontal_angle = np.mean([s.calc_h_angle() for s in squares])
    angle = horizontal_angle
    # print(f"Horizontal angle = {horizontal_angle}, rotate angle = {np.rad2deg(angle)}, squares count = {len(squares)}")

    h, w = image.shape[:2]
    center = (w//2, h//2)
    rotated_image = __rotate_image(image, angle, center=center)
    rotated_squares = __rotate_squares(squares, angle, center=center)

    return rotated_image, rotated_squares


def __rotate_image(image: MatLike, angle: float, scale=1.0, center=(0, 0)):
    h, w = image.shape[:2]
    
    M = cv2.getRotationMatrix2D(center, np.rad2deg(angle), scale)
    
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def __rotate_squares(squares: list[Square], angle: float, center=None):
    rotated_squares = []

    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    for square in squares:
        corners = square.approx
        if center is not None:
            corners = corners - center
    
        corners = np.dot(corners, rot_matrix)
        if center is not None:
            corners += center

        corners = corners.astype(np.int32)
        
        x, y = corners[0]
        rotated_square = Square(
            x=x,
            y=y,
            w=square.w,  # Note: width/height might need recalculation
            h=square.h,
            area=square.area,
            approx=corners
        )
        
        rotated_squares.append(rotated_square)
    
    return rotated_squares
