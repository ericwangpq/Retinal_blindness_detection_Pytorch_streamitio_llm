import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import warnings


class VisualizationService:
    """Enhanced service for handling visualization operations like GradCAM with accuracy improvements"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    def get_optimal_target_layer(self, model):
        """Intelligently select the best target layer based on model architecture"""
        # ResNet architecture
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        # VGG architecture
        elif hasattr(model, 'features'):
            return model.features[-1]
        # Models with backbone (e.g., some transfer learning setups)
        elif hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'layer4'):
                return model.backbone.layer4[-1]
        
        # Fallback - find the last convolutional layer
        last_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = module
        
        if last_conv_layer is None:
            raise ValueError("Could not find suitable target layer for GradCAM. Model must have at least one Conv2d layer.")
        
        return last_conv_layer
    
    def get_visualization_transforms(self, include_random_flip=False):
        """Get transforms for visualization with option to match training transforms exactly"""
        transforms = [torchvision.transforms.Resize((224, 224))]
        
        if include_random_flip:
            transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
        
        transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])
        return torchvision.transforms.Compose(transforms)
    
    def get_gradcam(self, model, img_tensor, target_layer, use_predicted_class=True, target_class=None):
        """Generate GradCAM visualization for the input image with correct target class"""
        cam = None
        try:
            cam = GradCAM(
                model=model,
                target_layers=[target_layer]
            )
            
            # Determine targets based on parameters
            targets = None  # Default: let GradCAM use highest probability class
            
            if not use_predicted_class and target_class is not None:
                # Use specific target class if provided
                targets = [ClassifierOutputTarget(target_class)]
            elif use_predicted_class:
                # Get model prediction to use predicted class
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor.to(self.device))
                    predicted_class = torch.argmax(output, dim=1).item()
                    targets = [ClassifierOutputTarget(predicted_class)]
            
            # Generate GradCAM
            grayscale_cam = cam(input_tensor=img_tensor.to(self.device), targets=targets)
            return grayscale_cam[0, :]
            
        except Exception as e:
            raise RuntimeError(f"GradCAM generation failed: {str(e)}")
        finally:
            # Clean up to prevent destructor issues
            if cam is not None:
                try:
                    if hasattr(cam, 'activations_and_grads'):
                        cam.activations_and_grads.release()
                except:
                    pass  # Ignore cleanup errors
    
    def generate_gradcam_visualization(self, model, uploaded_file, use_predicted_class=True):
        """Generate enhanced GradCAM visualization with 3 subplots and return as matplotlib figure"""
        # Ensure model is in evaluation mode
        model.eval()
        
        # Use consistent transforms (no random flip for visualization)
        test_transforms = self.get_visualization_transforms(include_random_flip=False)
        
        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = test_transforms(img).unsqueeze(0)
        
        # Get model prediction with confidence
        with torch.no_grad():
            output = model(input_tensor.to(self.device))
            probabilities = torch.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            predicted_class = top_class.item()
            confidence = top_p.item()
        
        # Get optimal target layer
        try:
            target_layer = self.get_optimal_target_layer(model)
        except ValueError as e:
            raise ValueError(f"GradCAM target layer selection failed: {str(e)}")
        
        # Generate GradCAM
        grayscale_cam = self.get_gradcam(model, input_tensor, target_layer, use_predicted_class)
        
        # Prepare image for visualization
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (224, 224))
        img_np = img_np / 255.0
        
        # Create visualization overlay using RGB color space
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        # Create enhanced figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Subplot 1: Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Subplot 2: Pure heatmap
        heatmap = plt.cm.jet(grayscale_cam)[:, :, :3]  # Convert to RGB, remove alpha
        axes[1].imshow(heatmap)
        axes[1].set_title('GradCAM Heatmap\n(Model Attention Areas)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Subplot 3: Overlaid visualization
        axes[2].imshow(visualization)
        prediction_text = f'Prediction: {self.classes[predicted_class]}\nConfidence: {confidence:.1%}'
        target_text = f'\n(Heatmap shows attention for predicted class)'
        axes[2].set_title(f'GradCAM Overlay\n{prediction_text}{target_text}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def generate_multi_layer_gradcam(self, model, uploaded_file):
        """Generate GradCAM for multiple layers to provide comprehensive analysis"""
        # Ensure model is in evaluation mode
        model.eval()
        
        # Use consistent transforms
        test_transforms = self.get_visualization_transforms(include_random_flip=False)
        
        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = test_transforms(img).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor.to(self.device))
            probabilities = torch.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            predicted_class = top_class.item()
            confidence = top_p.item()
        
        # Define target layers for multi-layer analysis (assuming ResNet)
        target_layers_info = []
        if hasattr(model, 'layer2') and hasattr(model, 'layer3') and hasattr(model, 'layer4'):
            target_layers_info = [
                (model.layer2[-1], 'Layer 2 (Low-level features)'),
                (model.layer3[-1], 'Layer 3 (Mid-level features)'),
                (model.layer4[-1], 'Layer 4 (High-level features)')
            ]
        else:
            # Fallback to single layer if not ResNet
            target_layer = self.get_optimal_target_layer(model)
            target_layers_info = [(target_layer, 'Target Layer')]
        
        # Prepare image for visualization
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (224, 224))
        img_np = img_np / 255.0
        
        # Create figure
        num_layers = len(target_layers_info)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(6 * (num_layers + 1), 6))
        
        # If only one subplot, make axes iterable
        if num_layers == 0:
            axes = [axes]
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Generate GradCAM for each layer
        for i, (target_layer, layer_name) in enumerate(target_layers_info):
            grayscale_cam = self.get_gradcam(model, input_tensor, target_layer, use_predicted_class=True)
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            axes[i + 1].imshow(visualization)
            axes[i + 1].set_title(f'{layer_name}\nGradCAM Analysis', fontsize=12, fontweight='bold')
            axes[i + 1].axis('off')
        
        # Add overall title
        fig.suptitle(f'Multi-Layer GradCAM Analysis\nPrediction: {self.classes[predicted_class]} ({confidence:.1%} confidence)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    

    
    def validate_model_compatibility(self, model):
        """Validate that the model is compatible with GradCAM analysis"""
        try:
            # Check if we can find a suitable target layer
            target_layer = self.get_optimal_target_layer(model)
            return True, f"Compatible - using target layer: {type(target_layer).__name__}"
        except ValueError as e:
            return False, str(e) 