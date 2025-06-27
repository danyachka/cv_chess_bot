from dataclasses import dataclass
from cv2.typing import MatLike
from enum import Enum
import numpy as np


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


