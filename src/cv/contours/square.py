import cv2
import numpy as np
from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Final

from src.cv.utils import calc_points_dist, calc_angle

rate: Final[float] = 0.15
square_area_percentage_threshold = (1 + rate) / (1 - rate)
square_area_percentage_threshold_group = (1 + rate) / (1 - rate)


@dataclass
class Square:
    x: int
    y: int
    w: int
    h: int

    area: float
    approx: MatLike # always 4 points only

    col_num: int = None
    row_num: int = None

    def calc_h_angle(self) -> float:
        return (calc_angle(self.approx[0], self.approx[3]) + calc_angle(self.approx[1], self.approx[2])) / 2

    def calc_side(self) -> float:
        return (self.w + self.h) / 2


def filter_squares(contours: MatLike) -> list[Square]:
    squares = []

    print(len(contours))
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < 300:
            continue

        approx = approx.reshape(4, 2)
        points, w, h = __recompose_square_points(approx)

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.85 or 1.15 < aspect_ratio:
            continue

        x, y = points[0]

        squares.append(Square(x, y, w, h, area, points))
    return squares



def cluster_squares(squares: list[Square]) -> list[list[Square]]:
    result = [squares]
    counter = 1
    while not __check_area_threshold_in_group(result[0]) and (len(result[0]) >= 3):
        print(f"Iterating clustering {counter}")
        counter += 1
        clustered = __cluster_group(result[0])
        result = [*clustered, *result[1:]]
    print("Resulting lens = ", *[len(item) for item in result])
    return result


def __check_area_threshold_in_group(squares: list[Square]) -> bool:
    areas = [square.w*square.h for square in squares]
    avg = np.mean(areas)

    for area in areas:
        if not __check_threshold(area, avg, square_area_percentage_threshold):
            return False
    return True


def __check_threshold(area1, area2, threshold) -> bool:
    return max(area1, area2) / min(area1, area2) < threshold**2


def __cluster_group(squares: list[Square]) -> list[list[Square]]:
    sides = np.zeros((len(squares), 2), dtype=np.float32)
    for i, square in enumerate(squares):
        sides[i, 0] = square.w 
        sides[i, 1] = square.h

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, average_vals = cv2.kmeans(sides, min(3, len(sides)), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    if not ret:
        print("Not successful squares kmeans search")
        return [squares]

    result = [(average_vals[i], []) for i in range(len(average_vals))]
    for i, val in enumerate(labels.ravel()):
        result[val][1].append(squares[i])
    result = sorted(result, key=lambda x: len(x[1]), reverse=True)
    print(
        ', '.join([str(i + 1) + ': ' + str(len(result[i])) for i in range(len(result))]),
        f", center = {average_vals}"
    )

    return __merge_similar_groups([item[1] for item in result], [item[0] for item in result])


def __merge_similar_groups(groups: list[list[Square]], average_vals) -> list[list[Square]]:
    areas = [center[0]*center[1] for center in average_vals]

    # assume that most of the squares are on chessboard
    result = [groups[0]]
    for i in range(1, len(groups)):
        if __check_threshold(areas[0], areas[i], square_area_percentage_threshold_group):
            result[0].extend(groups[i])
        else:
            result.append(groups[i])
    return result


# points, w, h
def __recompose_square_points(points):
    r = np.zeros_like(points)

    x_c = np.mean(points[:, 0])
    y_c = np.mean(points[:, 1])

    for point in points:
        x, y = point
        if x < x_c and y < y_c:
            r[0] = point
        elif x < x_c and y >= y_c:
            r[1] = point
        elif x >= x_c and y >= y_c:
            r[2] = point
        else:
            r[3] = point

    w = (calc_points_dist(r[3], r[0]) + calc_points_dist(r[2], r[1])) / 2
    h = (calc_points_dist(r[1], r[0]) + calc_points_dist(r[3], r[2])) / 2
    return r, w, h
