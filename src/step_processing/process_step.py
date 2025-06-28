from src.cv.chessboard.chessboard import Chessboard
from src.step_processing.chessboard_state import ChessboardState, start_fen


class StepProcessor:

    current_chessboard: Chessboard = None

    current_fen: ChessboardState = ChessboardState(start_fen)

