"""Clasifica fotos de perros y gatos con el modelo entrenado en explore.ipynb.

Uso:
    python src/app.py <imagen.jpg> [<imagen2.jpg> ...]
"""
import sys
from pathlib import Path

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dogs_vs_cats_efficientnetb0.keras"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["gato", "perro"]  # mismo orden que class_indices: {'cats': 0, 'dogs': 1}


def predict(model, image_paths):
    # EfficientNetB0 normaliza internamente: los píxeles se pasan en [0, 255]
    batch = np.stack([img_to_array(load_img(p, target_size=INPUT_SIZE)) for p in image_paths])
    return model.predict(batch, verbose=0)


def main():
    image_paths = sys.argv[1:]
    if not image_paths:
        sys.exit(__doc__)

    model = load_model(MODEL_PATH)
    for path, probs in zip(image_paths, predict(model, image_paths)):
        label = CLASS_NAMES[probs.argmax()]
        print(f"{path}: {label} ({probs.max():.1%})")


if __name__ == "__main__":
    main()
