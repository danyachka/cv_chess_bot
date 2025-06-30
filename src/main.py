from colorama import Fore
from stockfish import Stockfish
import argparse

from src.camera import select_camera
from src.cv.chessboard.chessboard import Chessboard
from src.cv.chessboard_find import find_chessboard
from src.step_processing.process_step import PlayingSide, StepProcessor


def main(elo: int):
    capture = select_camera()
    if capture is None:
        print(f"{Fore.RED}Exception: Can't start capture{Fore.RESET}")
        return

    player_side: str = None
    print("Select your side (w/b)")
    while player_side is None:
        s = input("-> ")
        if s == 'w' or s == 'b':
            player_side = s

    stockfish = Stockfish('/usr/games/stockfish')
    stockfish.set_elo_rating(elo)
    stepProcessor = StepProcessor(PlayingSide.WHITE if player_side == 'b' else PlayingSide.BLACK, stockfish)

    if stepProcessor.bot_playing_side == PlayingSide.WHITE:
        stepProcessor.make_bots_move()

    stepProcessor.current_fen.print()

    while True:
        s = input(f"""{Fore.GREEN}Print {Fore.MAGENTA}q{Fore.GREEN} to {Fore.MAGENTA}end game{Fore.GREEN} or {Fore.MAGENTA}any{Fore.GREEN} other letter when your move is made!{Fore.RESET}""")
        if s == 'q':
            return
        
        ret, frame = capture.read()
        if not ret:
            print(f"{Fore.RED}Exception: Can't read a picture{Fore.RESET}")
        
        ## chessboard
        new_chess_board: Chessboard = find_chessboard(frame, is_white_sided=stepProcessor.bot_playing_side==PlayingSide.WHITE, is_test=False)
        if new_chess_board is None:
            print(f"{Fore.RED}Exception: Can't find chessboard{Fore.RESET}")
            continue

        ## process step
        if not stepProcessor.process_enemy_step(new_chess_board):
            continue
        if stepProcessor.is_game_ended(False):
            break

        if not stepProcessor.make_bots_move():
            continue
        if stepProcessor.is_game_ended(True):
            break
        
    capture.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CV && Stockfish based chess bot')
    parser.add_argument('--elo', type=int, default=1350, help="Bot's elo")
    args = parser.parse_args()
    
    main(elo=args.elo)
