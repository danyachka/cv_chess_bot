import cv2
from cv2.typing import MatLike
from src.cv import utils
from dataclasses import dataclass


@dataclass
class Square:
   x: int 
   y: int
   w: int
   h: int

   area: float
   approx: MatLike


def find_chessboard(image: MatLike, is_test=False) -> MatLike:
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   edges = utils.get_edges(gray=gray)
   
   if is_test:
      utils.show_image(edges)

   contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   squares = __filter_squares(contours)
   
   if is_test:
      print(f"Found squares: {squares}")
      line_img = image.copy()
      # Draw 
      for square in squares:
         color = (255, 0, 0)#list(np.random.random(size=3) * 256)
         cv2.drawContours(line_img, [square.approx], 0, color, 2)

      utils.show_image(line_img)
   return None


def __filter_squares(contours) -> list[Square]:
   squares = []

   print(len(contours))
   for cnt in contours:
      approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

      if len(approx) > 6: 
         continue 
      if not cv2.isContourConvex(approx): 
         continue

      area = cv2.contourArea(approx)
      if area < 1500:
         continue

      (x, y, w, h) = cv2.boundingRect(approx)

      aspect_ratio = float(w) / h
      if aspect_ratio < 0.85 or 1.15 < aspect_ratio:
         continue
         
      squares.append(Square(x, y, w, h, area, approx))
   return squares