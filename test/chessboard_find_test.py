from src.cv.chessboard_find import find_chessboard
import cv2
from pathlib import Path

names = [
    # "new_2.jpg",
    "1.jpg",
    # "clear_0.jpg",
    # "clear_1.jpg"
]

for name in names:
    print(f"Handling {name}")
    path = Path().joinpath("data").joinpath(name)

    image = cv2.imread(path)

    wrapped = find_chessboard(image, is_white_sided=False, is_test=True)
    if wrapped is not None:
        image = wrapped
        print("wrapped has been gotten successfully")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("wrapped is null")

    # show_image(image=image)
