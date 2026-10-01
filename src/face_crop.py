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

def click_crop(image,x,y,size=400):
    half = size // 2

    left = max(0,x - half)
    top = max(0,y - half)
    right = min(image.width,x + half)
    bottom = min(image.height,y + half)
    
    if right - left < size:
        if left == 0:
            right = min(image.width,size)
        elif right == image.width:
            left = max(0,image.width - size)

    if bottom - top < size:
        if top == 0:
            bottom = min(image.height,size)
        elif bottom == image.height:
            top = max(0,image.height - size)

    crop = image.crop((left,top,right,bottom))
    return crop

def find_face_in_click_crop(crop,detector,confidence_threshold=0.90):

    crop_np = np.asarray(crop.convert("RGB"))
    detections = detector.detect_faces(crop_np)

    if not detections:
        return None

    best_face = max(detections,key=lambda face:face.get("confidence",0))
    confidence = float(best_face.get("confidence",0))

    if confidence < confidence_threshold:
        return None

    x, y, w, h = best_face["box"]

    padding_x = int(w * 0.15)
    padding_y = int(h * 0.15)

    x1 = max(x - padding_x,0)
    y1 = max(y - padding_y,0)
    x2 = min(x + w + padding_x,crop.width)
    y2 = min(y + h + padding_y,crop.height)
    
    face_crop = crop.crop((x1,y1,x2,y2))

    return { "image": face_crop,"source": "Click", "confidence": confidence}
