import cv2
from cv2.typing import MatLike

from src.cv import utils
from src.cv.contours.square import filter_squares, Square


def find_chessboard(image: MatLike, is_test=False) -> MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel = utils.process_sobel(gray)
    edges = utils.get_edges(gray=sobel, iterations=0)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = filter_squares(contours)

    if is_test:
        __show_test_images(image, edges, squares)

    return None


def __show_test_images(image, edges, squares: list[Square]):
    utils.show_image(edges)
    print(f"Found squares: {squares}")
    line_img = image.copy()
    # Draw
    for square in squares:
        color = (255, 0, 0)  # list(np.random.random(size=3) * 256)
        cv2.drawContours(line_img, [square.approx], 0, color, 2)

    utils.show_image(line_img)

