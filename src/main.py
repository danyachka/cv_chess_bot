import cv2

from src.camera import select_camera


def main():
    capture = select_camera()

    print("Enter 'q' to leave")

    while True:
        s = input("Print")
        if s == 'q':
            return
        
        ret, frame = capture.read()
        if not ret:
            print("Exception: can't read a picture")
        
        ## process step
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()

if __name__ == "__main__":
    main()
