import os
import cv2

def save_all_frames(video_path, dir_path, basename, resize = None, step = 1, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
          if n % step == 0:
            if resize is not None:
              frame = cv2.resize(frame, resize, interpolation = cv2.INTER_AREA)
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
          n += 1
        else:
            return
            
