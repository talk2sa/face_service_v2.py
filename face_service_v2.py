# --- DeepFace / TensorFlow 2.20 Compatibility Patch ---
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    import tensorflow as tf
    # Workaround for missing LocallyConnected2D in new keras versions
    from tensorflow.keras.layers import LocallyConnected2D
except Exception as e:
    print("TensorFlow/Keras patch applied:", e)
# -------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
from deepface import DeepFace
from PIL import Image
import requests, io, numpy as np, cv2

app = FastAPI(title="Fundra Unified Face Verification", version="2.0.0")

class OnboardRequest(BaseModel):
    selfie_urls: list[str]  # 3 frames
    id_url: str

class VerifyRequest(BaseModel):
    selfie_urls: list[str]  # 3 frames
    stored_face_url: str

def fetch_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=10)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)

def brightness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    return np.clip((mean / 255.0) * 100, 0, 100)

def clarity_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return np.clip((lap_var / 300.0) * 100, 0, 100)

def liveness_score(frames: list[np.ndarray]) -> float:
    variances = []
    for i in range(1, len(frames)):
        diff = np.mean(cv2.absdiff(frames[i], frames[i - 1]))
        variances.append(diff)
    score = np.mean(variances)
    return np.clip((score / 15.0) * 100, 0, 100)

def dual_model_similarity(img1, img2) -> float:
    try:
        result1 = DeepFace.verify(
            img1_path=img1, img2_path=img2,
            model_name="Facenet512", enforce_detection=False
        )
        result2 = DeepFace.verify(
            img1_path=img1, img2_path=img2,
            model_name="ArcFace", enforce_detection=False
        )
        score = ((1 - result1["distance"]) + (1 - result2["distance"])) / 2
        return float(np.clip(score * 100, 0, 100))
    except Exception as e:
        print("Error in dual_model_similarity:", e)
        return 0.0

@app.post("/face/onboard")
def face_onboard(payload: OnboardRequest):
    try:
        frames = [fetch_image(url) for url in payload.selfie_urls]
        idimg = fetch_image(payload.id_url)

        clarity_values = [clarity_score(f) for f in frames]
        best_idx = int(np.argmax(clarity_values))
        best_selfie = frames[best_idx]

        similarity = dual_model_similarity(best_selfie, idimg)
        live = liveness_score(frames)
        bright = brightness_score(best_selfie)
        clarity_val = clarity_values[best_idx]

        total = (similarity * 0.4) + (live * 0.3) + (bright * 0.15) + (clarity_val * 0.15)

        return {
            "context": "onboarding",
            "similarity": round(similarity, 2),
            "liveness": round(live, 2),
            "brightness": round(bright, 2),
            "clarity": round(clarity_val, 2),
            "total": round(total, 2),
            "decision": "auto_verified" if total >= 75 else "manual_review"
        }

    except Exception as e:
        print("Onboarding Error:", e)
        return {"error": str(e), "total": 0}

@app.post("/face/verify")
def face_verify(payload: VerifyRequest):
    try:
        frames = [fetch_image(url) for url in payload.selfie_urls]
        stored_face = fetch_image(payload.stored_face_url)

        clarity_values = [clarity_score(f) for f in frames]
        best_idx = int(np.argmax(clarity_values))
        best_selfie = frames[best_idx]

        similarity = dual_model_similarity(best_selfie, stored_face)
        live = liveness_score(frames)
        bright = brightness_score(best_selfie)
        clarity_val = clarity_values[best_idx]

        total = (similarity * 0.5) + (live * 0.3) + (bright * 0.1) + (clarity_val * 0.1)

        return {
            "context": "transaction",
            "similarity": round(similarity, 2),
            "liveness": round(live, 2),
            "brightness": round(bright, 2),
            "clarity": round(clarity_val, 2),
            "total": round(total, 2),
            "decision": "verified" if total >= 80 else "reject"
        }

    except Exception as e:
        print("Verification Error:", e)
        return {"error": str(e), "total": 0}
