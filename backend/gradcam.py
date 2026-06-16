import os
import cv2
import numpy as np
import torch
from predict import preprocess_image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)  # shape: [1, 1]
        self.model.zero_grad()

        # Single sigmoid output — backprop through it directly
        target_score = torch.sigmoid(output[0, 0])
        target_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap


def generate_gradcam(model, image_path, output_dir='heatmaps', device='cpu'):
    target_layer = model.efficientnet.conv_head
    gradcam = GradCAM(model, target_layer)
    input_tensor = preprocess_image(image_path).to(device)
    cam = gradcam.generate(input_tensor)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_heatmap{ext}")
    cv2.imwrite(output_path, overlay)
    return output_path


if __name__ == "__main__":
    print("Grad-CAM module ready.")