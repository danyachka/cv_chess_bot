from dataclasses import dataclass
import cv2
from cv2.typing import MatLike
from enum import Enum
import numpy as np

from src.cv import utils


class Position(Enum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2


@dataclass
class Chessboard:
    wrapped: MatLike
    mean_dx: int
    mean_dy: int

    positions: tuple[tuple[Position]]

    def corners_of(self, row, col) -> np.ndarray:
        return corners_of(self.mean_dx, self.mean_dy, row, col)
    
    def show_highlighted_squares(self, positions: list[tuple[int, int]]) -> None:
        print(positions)
        image = self.wrapped.copy()

        for i, j in positions:
            contours = [self.corners_of(i, j)]
            
            cv2.drawContours(image, contours, 0, (255, 0, 255), 2)
        utils.show_image(image, "Found positions")
    

def corners_of(mean_dx, mean_dy, row, col) -> np.ndarray:
    x = col * mean_dx
    y = (7 - row) * mean_dy
    return np.array(
        [
            [x, y],
            [x, y + mean_dy],
            [x + mean_dx, y + mean_dy],
            [x + mean_dx, y],
        ],
        dtype=np.int32,
    )


