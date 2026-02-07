import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # Traffic Sign Recognition/
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB" / "Train"
MODELS_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
# ---------------------------

def load_gtsrb_train(data_dir: Path, img_size: int):
    """
    Loads GTSRB data from folder-per-class structure:
      Train/0/*.png
      Train/1/*.png
      ...
      Train/42/*.png
    """
    X, y = [], []
    class_ids = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda p: int(p.name))

    if not class_ids:
        raise FileNotFoundError(f"No class folders found in: {data_dir}")

    for class_dir in class_ids:
        label = int(class_dir.name)
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y, [int(d.name) for d in class_ids]

def build_model(num_classes: int, img_size: int):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {DATA_DIR}")
    X, y, class_ids = load_gtsrb_train(DATA_DIR, IMG_SIZE)
    num_classes = len(class_ids)

    print(f"Loaded: X={X.shape}, y={y.shape}, classes={num_classes}")

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(num_classes=num_classes, img_size=IMG_SIZE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "sign_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model too
    model.save(MODELS_DIR / "sign_model_final.keras")

    # Save labels (class IDs)
    labels_path = MODELS_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"class_ids": class_ids}, f, indent=2)

    print("\n✅ Training done!")
    print(f"Saved best model to: {MODELS_DIR / 'sign_model.keras'}")
    print(f"Saved final model to: {MODELS_DIR / 'sign_model_final.keras'}")
    print(f"Saved labels to: {labels_path}")

if __name__ == "__main__":
    main()
