import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt


class VisualizationService:
    """Service for handling visualization operations like GradCAM"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_gradcam(self, model, img_tensor, target_layer):
        """Generate GradCAM visualization for the input image"""
        cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=img_tensor.to(self.device), targets=targets)
        return grayscale_cam[0, :]
    
    def get_visualization_transforms(self):
        """Get transforms for visualization (without random flip)"""
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])
    
    def generate_gradcam_visualization(self, model, uploaded_file):
        """Generate GradCAM visualization and return as matplotlib figure"""
        test_transforms = self.get_visualization_transforms()
        
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = test_transforms(img).unsqueeze(0)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(self.device))
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            predicted_class = top_class.item()
        
        # Get GradCAM
        target_layer = model.layer4[-1]
        grayscale_cam = self.get_gradcam(model, input_tensor, target_layer)
        
        # Convert image to numpy array for visualization
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (224, 224))
        img_np = img_np / 255.0
        
        # Create visualization
        visualization = show_cam_on_image(img_np, grayscale_cam)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(img_np)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # GradCAM visualization
        ax2.imshow(visualization)
        ax2.set_title(f'GradCAM Visualization\nPredicted Class: {predicted_class}')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig 