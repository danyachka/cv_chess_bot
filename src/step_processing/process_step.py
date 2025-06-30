from enum import Enum
from typing import Final

from colorama import Fore

from src.cv.chessboard.chessboard import Chessboard, Position
from src.step_processing.chessboard_state import ChessboardState, create_from_fen, start_fen
from src.step_processing.chess_piece import Piece, PieceType
from stockfish import Stockfish


cols_names: Final[str] = "abcdefgh"


class Move:
    name: str
    start: tuple[int, int]
    end: tuple[int, int]

    def __init__(self, start, end, name=None):
        self.start = start
        self.end = end
        if name is None:
            self.name = self.__get_step_name(start[0], start[1], end[0], end[1])
        else:
            self.name = name
    
    def __get_step_name(self, i1, j1, i2, j2) -> str:
        return f"{cols_names[j1]}{i1}{cols_names[j2]}{i2}"

class PlayingSide(Enum):
    WHITE = 0
    BLACK = 1


class StepProcessor:
    bot_playing_side: PlayingSide
    stockfish_board: Stockfish
    
    current_fen: ChessboardState = create_from_fen(start_fen)

    def __init__(self, playing_side: PlayingSide, stockfish_board: Stockfish):
        self.bot_playing_side = playing_side
        self.stockfish_board = stockfish_board

    def process_enemy_step(self, new_chessboard: Chessboard) -> bool:
        changed_positions = self.__find_changed_positions(new_chessboard)
        move = self.__find_move(changed_positions, new_chessboard)

        if move is None:
            print(f"{Fore.RED}Exception: Couldn't find move!{Fore.RESET}")
            new_chessboard.show_highlighted_squares(changed_positions)
            return False
        
        new_chessboard.show_highlighted_squares([move.start, move.end])
        print(f"{Fore.CYAN}Were move {move.name}? (y/n){Fore.RESET}")
        while True:
            s = input("-> ")
            if s == 'y':
                break
            elif s == 'n':
                return False
        
        if not self.stockfish_board.is_move_correct(move.name):
            print(f"{Fore.RED}Exception: Incorrect move ({move.name})! Come back to previous position:{Fore.RESET}")
            self.stockfish_board.get_board_visual()
            return False
        
        # Record player's move
        print(f"{Fore.CYAN}Nice! You've done move {move.name}{Fore.RESET}")
        self.stockfish_board.make_moves_from_current_position([move.name])
        self.current_fen = create_from_fen(self.stockfish_board.get_fen_position())
        self.stockfish_board.get_board_visual()

        return True
    
    def make_bots_move(self) -> bool:
        bot_move = self.stockfish_board.get_best_move_time(time=500)
        print(f"{Fore.BLUE}Bot's done move {bot_move}{Fore.RESET}")
        self.stockfish_board.make_moves_from_current_position([bot_move])
        self.current_fen = create_from_fen(self.stockfish_board.get_fen_position())
        self.stockfish_board.get_board_visual()

        return True
    
    def __find_move(self, changed_positions: list[tuple[int, int]], new_chessboard: Chessboard) -> Move:
        from_positions: list[tuple[int, int]] = []
        to_positions: list[tuple[int, int]] = []

        for i, j in changed_positions:
            old_piece: Piece = self.current_fen.grid[i][j]
            new_data: Position = new_chessboard.positions[i][j]

            if self.__is_enemy(new_data):
                to_positions.append((i, j))
            elif self.__was_enemy(old_piece):
                from_positions.append((i, j))

        ## In case simple move (1 figure moved)
        if len(from_positions) == 1 and len(to_positions) == 1:
            f = from_positions[0]
            to = to_positions[0]
            return Move(f, to)
        
        castling_move = self.__get_castling_move(from_positions, to_positions)
        if castling_move is not None:
            return castling_move
        
    def __find_changed_positions(self, new_chessboard: Chessboard) -> list[tuple[int, int]]:
        changed = []
        for i in range(8):
            for j in range(8):
                old_piece: Piece = self.current_fen.grid[i][j]
                new_data: Position = new_chessboard.positions[i][j]

                if not self.__is_changed(old_piece, new_data):
                    continue
                changed.append((i, j))
        return changed


    def __is_enemy(self, position: Position) -> bool:
        if self.bot_playing_side == PlayingSide.WHITE:
            return position == Position.BLACK
        else:
            return position == Position.WHITE
        
    def __was_enemy(self, piece: Piece) -> bool:
        if self.bot_playing_side == PlayingSide.WHITE:
            return not piece.white and piece.type != PieceType.EMPTY
        else:
            return piece.white and piece.type != PieceType.EMPTY


    def __is_changed(self, old_piece: Piece, new_position: Position) -> bool:
        match new_position:
            case Position.WHITE:
                return (not old_piece.white and old_piece.type != PieceType.EMPTY) or old_piece.type == PieceType.EMPTY
            case Position.BLACK:
                return (old_piece.white and old_piece.type != PieceType.EMPTY) or old_piece.type == PieceType.EMPTY
            case Position.EMPTY:
                return old_piece.type != PieceType.EMPTY

    def __get_castling_move(self, from_pos: list[tuple[int, int]], to_pos: list[tuple[int, int]]) -> Move | None:
        if (len(from_pos) != 2) or (len(to_pos) != 2):
            return None
        for _, col in [*from_pos, *to_pos]:
            if self.bot_playing_side == PlayingSide.WHITE:
                if col != 7:
                    return None
            else:
                if col != 0:
                    return None
        
        left_from = from_pos[0]
        right_from = from_pos[1]
        
        left_to = to_pos[0]
        right_to = to_pos[1]

        if (
            (left_from[1] == 0 and right_from[1] == 4) and
            (left_to[1] == 2 and right_to[1] == 3)
        ): # long castling
            return Move(right_from, left_to, "O-O-O")
        elif (
            (left_from[1] == 4 and right_from[1] == 7) and
            (left_to[1] == 5 and right_to[1] == 6)
        ): # short castling
            return Move(left_from, right_to, "O-O")
        return None
    
    def is_game_ended(self, is_bot) -> bool:
        game_result = self.__get_game_over_status()
        if game_result is None:
            return False
        print(f"{Fore.MAGENTA}{'Bot' if is_bot else 'You'} has ended game! {game_result.upper()}!{Fore.RESET}")
    
    def __get_game_over_status(self):
        evaluation = self.stockfish_board.get_evaluation()
        
        if evaluation["type"] == "mate" and evaluation["value"] == 0:
            return "checkmate"
        elif evaluation["type"] == "mate" and abs(evaluation["value"]) > 0:
            return "stalemate"
        else:
            return None

