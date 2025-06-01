import torch
from torch import nn
import torchvision
import torchvision.models as models
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np

class ModelService:
    """Service for handling model operations"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        
    def init_model(self):
        """Initialize the ResNet152 model architecture"""
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        out_ftrs = 5
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, out_ftrs),
            nn.LogSoftmax(dim=1)
        )
        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.00001
        )
        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        model.to(self.device)
        
        # Set requires_grad for specific layers
        for name, child in model.named_children():
            if name in ['layer2', 'layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
                    
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.000001
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        return model, optimizer
    
    def load_model(self, path):
        """Load trained model from checkpoint"""
        model, optimizer = self.init_model()
        
        # For PyTorch 2.6+ compatibility - use weights_only=False for trusted checkpoint
        print("⚠️  Loading model checkpoint (trusted source)")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model = model
        return model
    
    def get_transforms(self):
        """Get image preprocessing transforms"""
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])
    
    def predict_image(self, uploaded_file):
        """Make prediction on uploaded image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        test_transforms = self.get_transforms()
        
        file = Image.open(uploaded_file).convert('RGB')
        img = test_transforms(file).unsqueeze(0)
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(img.to(self.device))
            ps = torch.exp(out)
            top_p, top_class = ps.topk(1, dim=1)
            value = top_class.item()
            probability = top_p.item()
            
        return {
            "class": self.classes[value],
            "value": value,
            "probability": probability
        }
    
    def get_model(self):
        """Get the loaded model"""
        return self.model
    
    def get_device(self):
        """Get the device being used"""
        return self.device 