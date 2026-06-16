import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=False):
        super(DeepfakeDetector, self).__init__()

        self.efficientnet = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        self.classifier = nn.Sequential(
            nn.Linear(1792 + 384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x)
        combined = torch.cat([eff_features, vit_features], dim=1)
        return self.classifier(combined)


def get_model(device='cpu', weights_path=None):
    model = DeepfakeDetector(pretrained=False)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)

        # Handle all Lightning checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Detected PyTorch Lightning checkpoint — extracting state_dict")
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip 'model.' prefix if present
        if any(k.startswith('model.') for k in state_dict.keys()):
            print("Stripping 'model.' prefix from checkpoint keys")
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

        matched = sum(1 for k in state_dict if k in dict(model.named_parameters()))
        total = len(dict(model.named_parameters()))
        print(f"Keys matched: {matched}/{total}")

        if matched == 0:
            print("WARNING: No weights were loaded")

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")

    model.to(device)
    model.eval()
    return model