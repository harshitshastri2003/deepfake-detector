import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import get_model


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ['Real', 'Fake']


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = PREPROCESS(image).unsqueeze(0)
    return tensor


def predict(image_path, model, device='cpu'):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        'label': CLASS_NAMES[pred_idx.item()],
        'confidence': round(confidence.item() * 100, 2),
        'probabilities': {
            'Real': round(probs[0][0].item() * 100, 2),
            'Fake': round(probs[0][1].item() * 100, 2)
        }
    }


if __name__ == "__main__":
    print("Predict module ready.")