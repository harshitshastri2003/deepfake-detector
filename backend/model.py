import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """
    Hybrid model combining EfficientNet-B4 (CNN) and ViT (Transformer).
    EfficientNet captures local texture artifacts; ViT captures global inconsistencies.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeDetector, self).__init__()

        # EfficientNet-B4 backbone (CNN branch)
        self.efficientnet = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )
        efficientnet_features = self.efficientnet.num_features  # 1792

        # ViT-Base backbone (Transformer branch)
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        vit_features = self.vit.num_features  # 768

        # Fusion + classifier head
        combined_features = efficientnet_features + vit_features  # 2560
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features from both branches
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x)

        # Concatenate and classify
        combined = torch.cat([eff_features, vit_features], dim=1)
        output = self.classifier(combined)
        return output


def get_model(device='cpu', weights_path=None):
    """Load model, optionally with trained weights."""
    model = DeepfakeDetector(num_classes=2, pretrained=True)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")

    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device=device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [1, 2]
    print(f"Model ready on {device}")