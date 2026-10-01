import numpy as np
from mtcnn import MTCNN


def create_detector():
    return MTCNN()


def mtcnn_crop(image,detector,confidence_threshold=0.975,min_face_size=40,padding=0.15):

    image = image.convert("RGB")
    image_np = np.asarray(image)

    detections = detector.detect_faces(image_np)

    results = []

    for face in detections:

        confidence = float(
            face.get("confidence",0))

        if (confidence< confidence_threshold):
            continue

        x, y, w, h = face["box"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        if (w < min_face_size or h < min_face_size):
            continue


        padding_x = int(w * padding)
        padding_y = int(h * padding)
      
        x1 = max(x - padding_x,0)
        y1 = max(y - padding_y,0)
        x2 = min(x + w + padding_x,image.width)
        y2 = min(y + h + padding_y,image.height)

        if (x2 <= x1 or y2 <= y1):
            continue

        crop = image.crop((x1,y1,x2,y2))
        results.append({"image":crop,"source":"MTCNN","confidence": confidence,"box":
            (x1,y1,x2,y2)})

    return results
