from src.cv.chessboard.chessboard_position_check import define_position_type
import cv2
from pathlib import Path

names = ["cell_bb.png"]#, "cell_w.png", "cell_e.png", "cell_b.png"]#, "1.jpg", "clear_0.jpg", "clear_1.jpg"]

for name in names:
  print(f"Handling {name}")
  path = Path().joinpath("data").joinpath(name)

  image = cv2.imread(path)

  wrapped = define_position_type(image, is_black_cell=True, is_test=True)
  if wrapped is not None:
    image = wrapped
    print("wrapped has been gotten successfully")
  else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("wrapped is null")

  # show_image(image=image)

