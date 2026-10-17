"""
CNN-based classification of healthy vs. diseased plant leaves.

Pipeline: load -> label -> stratified split -> preprocess/augment -> build CNN
          -> train -> test -> save plots/metrics -> predict on unseen images.

Dataset layout (either works; label = "healthy" if the folder name contains
the word 'healthy', otherwise 'diseased'):

  A) dataset/healthy/*.jpg , dataset/diseased/*.jpg
  B) PlantVillage style  : dataset/Tomato___healthy/*.jpg ,
                           dataset/Tomato___Early_blight/*.jpg , ...

Usage:
  python src/leaf_disease_cnn.py train   --data_dir dataset --epochs 25
  python src/leaf_disease_cnn.py predict --image path/to/leaf.jpg
  python src/leaf_disease_cnn.py predict --image_dir path/to/unseen_leaves
"""
import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display; notebooks override this
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = 128
CLASS_NAMES = ["healthy", "diseased"]          # label 0 / 1
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ROOT = Path(__file__).resolve().parent.parent
RESULTS, MODELS = ROOT / "results", ROOT / "models"
MODEL_PATH = MODELS / "leaf_cnn.keras"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------
# 1. DATA LOADING + SPLITTING
# --------------------------------------------------------------------------
def collect_images(data_dir):
    """Return (paths, labels). Label 0 = healthy, 1 = diseased."""
    paths, labels = [], []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            folder = p.parent.name.lower()
            paths.append(str(p))
            labels.append(0 if "healthy" in folder else 1)
    if not paths:
        raise FileNotFoundError(f"No images found under '{data_dir}'.")
    return np.array(paths), np.array(labels)


def split_data(paths, labels, val=0.15, test=0.15):
    """Stratified 70/15/15 split so class ratios are preserved."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=val + test, stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=test / (val + test), stratify=y_tmp,
        random_state=SEED)
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)


# --------------------------------------------------------------------------
# 2. PREPROCESSING  (resize -> scale to [0,1]; augmentation only for training)
# --------------------------------------------------------------------------
def load_and_preprocess(path, label, img_size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((img_size, img_size, 3))
    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels, batch=32, training=False, img_size=IMG_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(lambda p, l: load_and_preprocess(p, l, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def augmentation_block():
    """Random transforms; Keras applies these only while training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


# --------------------------------------------------------------------------
# 3. MODEL
# --------------------------------------------------------------------------
def build_cnn(img_size=IMG_SIZE):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = augmentation_block()(inputs)
    for filters in (32, 64, 128, 128):                 # 4 conv blocks
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(diseased)
    model = keras.Model(inputs, outputs, name="leaf_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])
    return model


# --------------------------------------------------------------------------
# 4. PLOTS
# --------------------------------------------------------------------------
def plot_history(history, out=RESULTS / "training_curves.png"):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_confusion(cm, out=RESULTS / "confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_sample_predictions(model, paths, labels, n=12,
                            out=RESULTS / "sample_predictions.png"):
    idx = np.random.RandomState(SEED).choice(
        len(paths), min(n, len(paths)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), idx):
        img, _ = load_and_preprocess(paths[i], labels[i])
        prob = float(model.predict(img[None], verbose=0)[0][0])
        pred = int(prob >= 0.5)
        conf = prob if pred else 1 - prob
        a.imshow(img.numpy())
        a.set_title(
            f"True: {CLASS_NAMES[labels[i]]}\n"
            f"Pred: {CLASS_NAMES[pred]} ({conf:.0%})",
            color="green" if pred == labels[i] else "red", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# 5. TRAIN + TEST
# --------------------------------------------------------------------------
def train(data_dir, epochs=25, batch=32, img_size=IMG_SIZE):
    set_seed()
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    paths, labels = collect_images(data_dir)
    print(f"Images: {len(paths)} | healthy={np.sum(labels == 0)} "
          f"diseased={np.sum(labels == 1)}")
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = split_data(paths, labels)
    print(f"Split -> train {len(p_tr)} | val {len(p_va)} | test {len(p_te)}")

    train_ds = make_dataset(p_tr, y_tr, batch, True, img_size)
    val_ds = make_dataset(p_va, y_va, batch, False, img_size)
    test_ds = make_dataset(p_te, y_te, batch, False, img_size)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    model = build_cnn(img_size)
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss",
                                        save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=callbacks)
    plot_history(history)

    # ---- testing on held-out images the model never saw ----
    probs = model.predict(test_ds, verbose=0).ravel()
    y_pred = (probs >= 0.5).astype(int)
    loss, acc, prec, rec = model.evaluate(test_ds, verbose=0)[:4]
    cm = confusion_matrix(y_te, y_pred)
    plot_confusion(cm)
    report = classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True)
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))
    metrics = {"test_loss": float(loss), "test_accuracy": float(acc),
               "test_precision": float(prec), "test_recall": float(rec),
               "roc_auc": float(roc_auc_score(y_te, probs)),
               "confusion_matrix": cm.tolist(), "report": report,
               "n_train": int(len(p_tr)), "n_val": int(len(p_va)),
               "n_test": int(len(p_te)),
               "epochs_run": len(history.history["loss"])}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_sample_predictions(model, p_te, y_te)
    print(f"Test accuracy: {acc:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Model saved to {MODEL_PATH}; plots/metrics in {RESULTS}")
    return model, history, metrics


# --------------------------------------------------------------------------
# 6. PREDICTION ON UNSEEN IMAGES
# --------------------------------------------------------------------------
def predict_image(model, image_path, img_size=IMG_SIZE, show=False,
                  save_to=None):
    img, _ = load_and_preprocess(image_path, 0, img_size)
    prob = float(model.predict(img[None], verbose=0)[0][0])
    pred = int(prob >= 0.5)
    conf = prob if pred else 1 - prob
    label = CLASS_NAMES[pred]
    print(f"{os.path.basename(image_path)} -> {label.upper()} "
          f"(confidence {conf:.1%})")
    if show or save_to:
        plt.figure(figsize=(4, 4))
        plt.imshow(img.numpy()); plt.axis("off")
        plt.title(f"{label.upper()} ({conf:.1%})",
                  color="green" if pred == 0 else "red")
        if save_to:
            plt.savefig(save_to, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
    return label, conf


def predict_cli(image=None, image_dir=None):
    model = keras.models.load_model(MODEL_PATH)
    files = []
    if image:
        files.append(image)
    if image_dir:
        files += [str(p) for p in sorted(Path(image_dir).iterdir())
                  if p.suffix.lower() in IMG_EXT]
    if not files:
        raise SystemExit("Provide --image or --image_dir")
    (ROOT / "screenshots").mkdir(exist_ok=True)
    for i, f in enumerate(files):
        predict_image(model, f,
                      save_to=ROOT / "screenshots" / f"prediction_{i + 1}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Healthy vs diseased leaf CNN")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data_dir", default=str(ROOT / "dataset"))
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--batch", type=int, default=32)
    pr = sub.add_parser("predict")
    pr.add_argument("--image")
    pr.add_argument("--image_dir")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data_dir, a.epochs, a.batch)
    else:
        predict_cli(a.image, a.image_dir)
