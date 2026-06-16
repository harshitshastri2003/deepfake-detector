import torch
from torchvision import transforms
from PIL import Image

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return PREPROCESS(image).unsqueeze(0)

def predict(image_path, model, device='cpu'):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(image_tensor).squeeze(1)
        prob = torch.sigmoid(logits).item()

    # class_to_idx = {'fake': 0, 'real': 1}
    # sigmoid > 0.5 → Real (label 1), sigmoid < 0.5 → Fake (label 0)
    prob_real = prob
    prob_fake = 1 - prob

    print(f"prob_fake: {prob_fake:.4f} | prob_real: {prob_real:.4f}")

    label = "Fake" if prob_fake > 0.5 else "Real"
    return {
        'label': label,
        'confidence': round(max(prob_fake, prob_real) * 100, 2),
        'probabilities': {
            'Real': round(prob_real * 100, 2),
            'Fake': round(prob_fake * 100, 2)
        }
    }