from enum import Enum
from dataclasses import dataclass

class PieceType(Enum):
    EMPTY = '*'
    PAN = 'P'
    ROOK = 'R'
    KNIGHT = 'N'
    BISHOP = 'B'
    KING = 'K'
    QUEEN = 'Q'


@dataclass
class Piece:
    type: PieceType
    white: bool

    def __str__(self) -> str:
        if self.white:
            self.type.name.upper()
        else:
            self.type.name.lower()
