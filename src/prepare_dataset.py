"""
Build the training set (and the held-out crops) from the PlantVillage 'color' folder.

Model 2 setup used in the report:
  * 11 crops for training: at most 300 images per healthy class, 150 per diseased class
  * 3 crops kept out of training completely (Cherry, Strawberry, Pepper) -> held-out test,
    40 images per class

Usage:
  python src/prepare_dataset.py --src PlantVillage-Dataset/raw/color --out dataset --heldout heldout

Note: the selection uses a fixed seed, but file order can differ between systems, so the
exact images may differ from the original Colab run.
"""
import argparse
import os
import random
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png")


def pick(src, cls, n, dst, rng):
    files = sorted(f for f in os.listdir(os.path.join(src, cls))
                   if f.lower().endswith(IMG_EXT))
    os.makedirs(os.path.join(dst, cls), exist_ok=True)
    for f in rng.sample(files, min(n, len(files))):
        shutil.copy(os.path.join(src, cls, f), os.path.join(dst, cls, f))


def main():
    ap = argparse.ArgumentParser(description="Prepare PlantVillage subsets")
    ap.add_argument("--src", required=True, help="PlantVillage 'color' folder")
    ap.add_argument("--out", default="dataset", help="training folder to create")
    ap.add_argument("--heldout", default="heldout", help="held-out crops folder to create")
    ap.add_argument("--held_crops", nargs="+", default=["Cherry", "Strawberry", "Pepper"])
    ap.add_argument("--healthy_cap", type=int, default=300)
    ap.add_argument("--diseased_cap", type=int, default=150)
    ap.add_argument("--held_per_class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    for d in (a.out, a.heldout):
        shutil.rmtree(d, ignore_errors=True)
    held = tuple(a.held_crops)
    for cls in sorted(os.listdir(a.src)):
        if "background" in cls.lower() or not os.path.isdir(os.path.join(a.src, cls)):
            continue
        if cls.startswith(held):
            pick(a.src, cls, a.held_per_class, a.heldout, rng)
        else:
            cap = a.healthy_cap if "healthy" in cls.lower() else a.diseased_cap
            pick(a.src, cls, cap, a.out, rng)

    count = lambda d: sum(len(f) for _, _, f in os.walk(d))
    print(f"Training images: {count(a.out)} | classes: {len(os.listdir(a.out))}")
    print(f"Held-out images: {count(a.heldout)}")


if __name__ == "__main__":
    main()
