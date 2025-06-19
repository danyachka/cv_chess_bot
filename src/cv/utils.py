import cv2
import numpy as np
from cv2.typing import MatLike


def get_edges(gray: MatLike):
   edges = gray

   _, edges = cv2.threshold(edges, 110, 255, cv2.THRESH_BINARY)

   edges = cv2.GaussianBlur(edges, (3, 3), 0)
   edges = cv2.Canny(edges, 60, 140, None, 3)

   edges = increase_lines_thickness(edges=edges)
   
   return edges


def increase_lines_thickness(edges):
   kernel = np.ones((3, 3), np.uint8)
   thickened = cv2.dilate(edges, kernel, iterations=1)

   return cv2.morphologyEx(thickened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def show_image(image, tag="Image"):
   cv2.imshow(tag, image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()