import torch
from torchvision import transforms


AGE_LABELS = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","more than 70"]
RACE_LABELS = ["Black","East Asian","Indian","Latino_Hispanic","Middle Eastern","Southeast Asian","White"]
GENDER_LABELS = ["Female","Male"]


transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])


def classify_faces(faces,model,device):

    if len(faces) == 0:
        return []

    tensors = []

    for face in faces:
        tensor = transformer(face["image"].convert("RGB"))
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():

        age_logits, race_logits, gender_logits = model(batch)

        age_probs = torch.sigmoid(age_logits)
        age_ids = (age_probs > 0.5).sum(dim=1)
      
        race_probs = torch.softmax(race_logits,dim=1)
        race_ids = torch.argmax(race_probs,dim=1)

        gender_probs = torch.sigmoid(gender_logits).view(-1)

        gender_ids = (gender_probs > 0.5).long()

    results = []

    for i, face in enumerate(faces):
      
        age_id = int(age_ids[i].item())
        race_id = int(race_ids[i].item())
        gender_id = int(gender_ids[i].item())

        result = {**face,"age":AGE_LABELS[age_id],"race":RACE_LABELS[race_id],"gender":GENDER_LABELS[gender_id]}
        results.append(result)
      
    return results


def filter_results(results,age_filter=None,race_filter=None,gender_filter=None):

    age_filter = (age_filter or [])
    race_filter = (race_filter or [])
    gender_filter = (gender_filter or [])

    filtered = []

    for result in results:

        if age_filter:
            if (result["age"] not in age_filter):
                continue

        if race_filter:

            if (result["race"] not in race_filter):
                continue

        if gender_filter:
            if (result["gender"]not in gender_filter):
                continue

        filtered.append(result)
    return filtered
