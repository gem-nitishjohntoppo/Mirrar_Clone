import cv2


def camera_input_live():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()

        if ret:
            # Convert the captured frame to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        else:
            break

    cap.release()
