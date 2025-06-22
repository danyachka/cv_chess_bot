import cv2
from cv2.typing import MatLike
from dataclasses import dataclass


@dataclass
class Square:
    x: int
    y: int
    w: int
    h: int

    area: float
    approx: MatLike


def filter_squares(contours: MatLike) -> list[Square]:
    squares = []

    print(len(contours))
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) > 6:
            continue
        if not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < 300:
            continue

        (x, y, w, h) = cv2.boundingRect(approx)

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.85 or 1.15 < aspect_ratio:
            continue

        squares.append(Square(x, y, w, h, area, approx))
    return squares