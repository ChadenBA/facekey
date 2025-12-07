import torch
from app.models.emotion_model import EmotionModel


def load_emotion_model(path, device="cpu"):
    """
    Charge un modèle d’émotion entraîné et le prépare pour l’inférence.
    - path : chemin du fichier .pth contenant les poids
    - device : 'cpu' ou 'cuda'
    """

    # 1️⃣ Créer une instance du modèle avec la même architecture que celle utilisée au training.
    # ⚠️ IMPORTANT : le modèle doit avoir la même structure que celle sauvegardée.
    model = EmotionModel()

    # 2️⃣ Charger les poids du réseau depuis le fichier sauvegardé.
    #    ➤ map_location permet d’utiliser un CPU même si le modèle a été entraîné avec GPU.
    #    ➤ weights_only=False car PyTorch 2.6 change le comportement par défaut.
    #      Ici on force le chargement normal.
    state_dict = torch.load(path, map_location=device, weights_only=False)

    # 3️⃣ Injecter le dictionnaire de poids dans l’architecture initialisée.
    #    ⚠️ Cela échouera si l’architecture ne correspond pas exactement aux poids.
    model.load_state_dict(state_dict)

    # 4️⃣ Déplacer le modèle vers CPU ou GPU en fonction de device.
    model.to(device)

    # 5️⃣ Passer en mode évaluation :
    #    - désactive dropout
    #    - désactive batch norm training
    #    - optimise inference
    model.eval()

    # 6️⃣ Retourner le modèle prêt à être utilisé
    return model
