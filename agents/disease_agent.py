"""
Disease Detection Agent — plain Python functions, compatible with google-adk FunctionTool.
CV model is lazy-loaded; app starts cleanly without the .pth file.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("agents.disease")
MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_labels: dict = {}
_transform = None


def _load_model() -> bool:
    global _model, _labels, _transform
    if _model is not None:
        return True
    model_path  = MODEL_DIR / "disease_model.pth"
    labels_path = MODEL_DIR / "disease_labels.json"
    if not model_path.exists():
        logger.warning("Disease model not found at %s — train in Colab first", model_path)
        return False
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision import models
        with open(labels_path) as f:
            _labels = json.load(f)
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(_labels))
        m.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        m.eval()
        _model = m
        _transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        logger.info("Disease model loaded (%d classes)", len(_labels))
        return True
    except Exception as e:
        logger.error("Failed to load disease model: %s", e)
        return False


_TREATMENTS = {
    "Healthy":                    {"organic": "No action needed.", "chemical": "None required.", "prevention": "Maintain regular watering and balanced fertilisation."},
    "Tomato___Late_blight":       {"organic": "Copper hydroxide spray; remove infected tissue.", "chemical": "Chlorothalonil or mancozeb every 5–7 days.", "prevention": "Avoid overhead watering; plant resistant varieties."},
    "Tomato___Early_blight":      {"organic": "Neem oil or copper spray. Remove lower infected leaves.", "chemical": "Azoxystrobin or chlorothalonil fungicide.", "prevention": "Mulch around base; rotate crops annually."},
    "Tomato___Bacterial_spot":    {"organic": "Copper-based bactericide spray.", "chemical": "Copper hydroxide + mancozeb tank mix.", "prevention": "Use certified disease-free seed; avoid working in wet fields."},
    "Corn_(maize)___Common_rust_":{"organic": "Remove heavily infected leaves.", "chemical": "Propiconazole or azoxystrobin at first sign.", "prevention": "Plant resistant hybrids; avoid late sowing."},
    "Corn_(maize)___Northern_Leaf_Blight": {"organic": "Crop rotation; remove debris after harvest.", "chemical": "Triazole fungicides at VT/R1 growth stage.", "prevention": "Use resistant varieties; reduce plant density."},
    "Potato___Late_blight":       {"organic": "Copper sulfate spray. Destroy infected plants.", "chemical": "Metalaxyl + mancozeb or cymoxanil.", "prevention": "Use certified seed; hill up soil around stems."},
    "Potato___Early_blight":      {"organic": "Neem oil spray; remove infected lower leaves.", "chemical": "Chlorothalonil or mancozeb every 7–10 days.", "prevention": "Avoid overhead irrigation; ensure adequate potassium."},
    "Rice___Leaf_Blast":          {"organic": "Silicon fertiliser strengthens cell walls.", "chemical": "Tricyclazole or isoprothiolane at booting stage.", "prevention": "Avoid excess nitrogen; maintain field drainage."},
    "Rice___Brown_spot":          {"organic": "Balanced NPK nutrition; silicon application.", "chemical": "Edifenphos or propiconazole fungicide.", "prevention": "Use healthy seed; maintain proper water management."},
    "Wheat___Yellow_Rust":        {"organic": "Remove volunteer wheat plants; improve drainage.", "chemical": "Propiconazole or tebuconazole at first sign.", "prevention": "Plant resistant varieties; avoid dense sowing."},
}
_DEFAULT_TREATMENT = {
    "organic": "Remove visibly infected leaves and improve air circulation.",
    "chemical": "Consult local agricultural extension officer for specific fungicide.",
    "prevention": "Maintain good crop hygiene; rotate crops annually.",
}


def detect_crop_disease(image_path: str) -> dict:
    """Detect crop disease from a local image file and recommend treatment.

    Args:
        image_path: Full path to the crop image file (jpg, png, etc.)

    Returns:
        Dictionary with detected disease, confidence score, treatment options, and urgency flag.
    """
    if not _load_model():
        return {
            "error": "Disease model not available. Train it in Colab first.",
            "tip": "Run scripts/train_disease_model_colab.py in Google Colab, then download disease_model.pth to models/",
        }
    try:
        import torch
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        x = _transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(_model(x), dim=1)[0]
            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])
        disease = _labels.get(str(top_idx), "Unknown")
        treatment = _TREATMENTS.get(disease, _DEFAULT_TREATMENT)
        return {
            "disease": disease,
            "confidence": round(confidence, 3),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "treatment": treatment,
            "urgent": confidence > 0.75 and disease != "Healthy",
            "note": "Confidence below 60% — consider sending sample to KVK for lab analysis." if confidence < 0.6 else "",
        }
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}
    except Exception as e:
        logger.error("Disease detection error: %s", e)
        return {"error": str(e)}


def get_disease_info(disease_name: str) -> dict:
    """Get treatment information for a known crop disease by name.

    Args:
        disease_name: Name of the disease (e.g. 'Tomato Late Blight', 'Rice Blast')

    Returns:
        Treatment options including organic, chemical, and prevention methods.
    """
    query = disease_name.lower().replace(" ", "_").replace("-", "_")
    for key, treatment in _TREATMENTS.items():
        if query in key.lower() or key.lower() in query:
            return {"disease": key, "treatment": treatment, "source": "FasalSetu disease database"}
    return {
        "disease": disease_name,
        "treatment": _DEFAULT_TREATMENT,
        "note": f"'{disease_name}' not found in database. Showing general advice.",
        "suggestion": "Consult your local Krishi Vigyan Kendra (KVK) for lab diagnosis.",
    }
