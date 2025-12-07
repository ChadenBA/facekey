#emotion_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class EmotionModel(nn.Module):
    """
    Modèle de reconnaissance d'émotions basé sur EfficientNet-B0.
    
    Architecture sans wrapper pour compatibilité avec les poids sauvegardés.
    Les layers sont directement accessibles : features, avgpool, classifier.
    """
    
    def __init__(self, num_classes=7):
        super().__init__()

        # Charger EfficientNet-B0 pré-entraîné
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        efficientnet = models.efficientnet_b0(weights=weights)

        # ✅ Copier les layers DIRECTEMENT (sans self.model wrapper)
        # Cela évite le préfixe "model." dans les clés du state_dict
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Modifier le classifier pour 7 émotions
        in_features = efficientnet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        # Transformation preprocessing pour serveur (224x224)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Labels des émotions
        self.labels = [
            "Angry", "Disgust", "Fear", "Happy", 
            "Sad", "Surprise", "Neutral"
        ]

    def forward(self, x):
        """
        Forward pass à travers le réseau.
        
        Args:
            x: Tensor de forme (batch, 3, 224, 224)
        
        Returns:
            Tensor de forme (batch, num_classes) avec les logits
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, image_pil, device="cpu"):
        """
        Prédit l'émotion depuis une image PIL.
        
        Args:
            image_pil: Image PIL (n'importe quelle taille, sera redimensionnée)
            device: "cpu" ou "cuda"
        
        Returns:
            tuple: (pred_idx, pred_label, probs_dict)
                - pred_idx: Index de la classe prédite (0-6)
                - pred_label: Nom de l'émotion ("Happy", "Sad", etc.)
                - probs_dict: Dict {emotion: probabilité} pour toutes les classes
        
        Example:
            >>> model = EmotionModel()
            >>> img = Image.open("face.jpg")
            >>> idx, label, probs = model.predict(img)
            >>> print(f"Émotion: {label} ({probs[label]:.2%})")
        """
        # Forcer mode inference
        self.eval()

        # Sécuriser le format de l'image
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # Préprocessing
        tensor = self.preprocess(image_pil).unsqueeze(0).to(device)

        # Forward pass sans gradients
        with torch.no_grad():
            output = self.forward(tensor)

        # Calculer les probabilités
        probs = torch.softmax(output, dim=1)[0]  # Vecteur 1D

        # Prédiction
        pred_idx = torch.argmax(probs).item()
        pred_label = self.labels[pred_idx]

        # Créer le dictionnaire de probabilités
        probs_dict = {
            self.labels[i]: float(probs[i]) 
            for i in range(len(self.labels))
        }

        return pred_idx, pred_label, probs_dict
    
    def predict_batch(self, images_pil, device="cpu"):
        """
        Prédit les émotions pour un batch d'images.
        
        Args:
            images_pil: Liste d'images PIL
            device: "cpu" ou "cuda"
        
        Returns:
            list: Liste de tuples (pred_idx, pred_label, probs_dict)
        """
        self.eval()
        
        # Préprocesser toutes les images
        tensors = []
        for img in images_pil:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = self.preprocess(img)
            tensors.append(tensor)
        
        # Stack en batch
        batch = torch.stack(tensors).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(batch)
        
        # Calculer les probabilités
        probs_batch = torch.softmax(outputs, dim=1)
        
        # Extraire les prédictions
        results = []
        for probs in probs_batch:
            pred_idx = torch.argmax(probs).item()
            pred_label = self.labels[pred_idx]
            probs_dict = {
                self.labels[i]: float(probs[i]) 
                for i in range(len(self.labels))
            }
            results.append((pred_idx, pred_label, probs_dict))
        
        return results   