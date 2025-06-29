from dataclasses import dataclass
from typing import Final

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
    
    def to_fen(self) -> str:
        fen = []

        for i in range(len(self.grid)):
            empty_counter = 0
            row: tuple[Piece] = self.grid[7 - i]

            for j in range(8):
                char = str(row[j])

                if char == '*':
                    empty_counter += 1
                else:
                    if empty_counter != 0:
                        fen.append(str(empty_counter))
                        empty_counter = 0
                    fen.append(char)
            if empty_counter != 0:
                fen.append(str(empty_counter))
            if i != 7:
              fen.append('/')
        
        fen.append(" ")
        fen.append('w' if self.is_white_step_side else 'b')

        fen.append(" ")
        fen.append(self.castling)

        fen.append(" ")
        fen.append(self.coords_pan_did_long_step)

        fen.append(" ")
        fen.append(str(self.draw_counter))

        fen.append(" ")
        fen.append(str(self.step_num))

        return ''.join(fen)


def create_from_fen(fen: str) -> ChessboardState:
    grid = [[Piece(PieceType.EMPTY, False) for i in range(8)] for j in range(8)]

    array = fen.split(' ')

    positions = array[0].replace("/", "")

    pos = 0
    for i in range(len(positions)):
        row = 7 - pos//8
        col = pos % 8

        char = positions[i]

        if char.isdigit():
            pos += int(char)
        elif char == '/':
            continue
        else:
            piece = PieceType(char.upper())
            grid[row][col] = Piece(piece, char.isupper())

    # step side
    white = array[1] == 'w'
    
    castling = array[2] if array[2] != "-" else ''

    coords_pan_did_long_step = array[3]

    draw_counter = int(array[4])

    step = int(array[-1])

    return ChessboardState(
        grid=((p for p in row) for row in grid),
        castling=castling,
        is_white_step_side=white,
        coords_pan_did_long_step=coords_pan_did_long_step,
        draw_counter=draw_counter,
        step_num=step
    )
