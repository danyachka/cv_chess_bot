import cv2


def select_camera():
    cameras = list_cameras()
    
    if len(cameras) == 0:
        print("No available camera has been found!")
        return
    
    print("Available cameras:")
    for cam in cameras:
        print(f"- Cam {cam[1]}")

    cap = None
    while cap is None:
        try:
            selection = int(input("Select camera: "))
            for cam in cameras:
                if cam[0] != selection:
                    continue
                cap = cv2.VideoCapture(selection)
            print("Exception: can't find this camera")
        except ValueError:
            print("Exception: enter number")

    if not cap.isOpened():
        print("Exception: can't open this camera")
        return
    return cap
    


def list_cameras(max_to_check=5) -> list[tuple[int, str]]:
    available = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append((i, cap.getBackendName))
            cap.release()
    return available

