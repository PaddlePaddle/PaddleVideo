import base64
import json

import cv2
import numpy as np
import requests


def numpy_to_base64(array: np.ndarray) -> str:
    """numpy_to_base64

    Args:
        array (np.ndarray): input ndarray.

    Returns:
        bytes object: encoded str.
    """
    return base64.b64encode(array).decode('utf8')


def video_to_numpy(file_path: str) -> np.ndarray:
    """decode video with cv2 and return stacked frames
       as numpy.

    Args:
        file_path (str): video file path.

    Returns:
        np.ndarray: [T,H,W,C] in uint8.
    """
    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret is False:
            continue
        img = frame[:, :, ::-1]
        decoded_frames.append(img)
    decoded_frames = np.stack(decoded_frames, axis=0)
    return decoded_frames


if __name__ == "__main__":
    url = "http://127.0.0.1:18080/video/prediction"

    video_path = '../../data/example.avi'

    # decoding video and get stacked frames as ndarray
    decoded_frames = video_to_numpy(file_path=video_path)

    # encode ndarray to base64 string for transportation.
    decoded_frames_base64 = numpy_to_base64(decoded_frames)

    # generate dict & convert to json.
    data = {
        "key": ["frames", "frames_shape"],
        "value": [decoded_frames_base64,
                  str(decoded_frames.shape)]
    }
    data = json.dumps(data)

    # transport to server & get get results.
    for i in range(1):
        r = requests.post(url=url, data=data, timeout=100)
        print(r.json())
