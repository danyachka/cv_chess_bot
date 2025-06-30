from dataclasses import dataclass
from typing import Final

from colorama import Fore

from src.step_processing.chess_piece import Piece, PieceType


start_fen: Final[str] = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

@dataclass
class ChessboardState:
    grid: tuple[tuple[Piece]]

    castling: str

    is_white_step_side: bool

    coords_pan_did_long_step: str

    draw_counter: int

    step_num: int

    def print(self):
        for row in self.grid:
            print(*[f"{Fore.BLUE if not p.white else ''}{p.type.name}{Fore.RESET}" for p in row])

    def can_castle(self, white: bool, short_side: bool) -> bool:
        if white:
            if short_side:
                return 'Q' in self.castling
            else:
                return 'K' in self.castling
        else:
            if short_side:
                return 'q' in self.castling
            else:
                return 'k' in self.castling


def create_from_fen(fen: str) -> ChessboardState:
    grid = [[Piece(PieceType.EMPTY, False) for i in range(8)] for j in range(8)]

    array = fen.split(' ')

    positions = array[0]
    pos = 0
    row = 7
    col = 0
    for i in range(len(positions)):
        char = positions[i]

        if char.isdigit():
            n = int(char)
            pos += n
            col += n
        elif char == '/':
            row -= 1
            col = 0
            continue
        else:
            piece = PieceType(char.upper())
            grid[row][col] = Piece(piece, char.isupper())
            col += 1

    # step side
    white = array[1] == 'w'
    
    castling = array[2] if array[2] != "-" else ''

    coords_pan_did_long_step = array[3]

    draw_counter = int(array[4])

    step = int(array[-1])        

    return ChessboardState(
        grid=tuple(tuple(p for p in row) for row in grid),
        castling=castling,
        is_white_step_side=white,
        coords_pan_did_long_step=coords_pan_did_long_step,
        draw_counter=draw_counter,
        step_num=step
    )
