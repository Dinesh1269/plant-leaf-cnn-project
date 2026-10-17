# Plant Leaf Disease Classification using CNN

CNN (TensorFlow/Keras) that classifies a plant leaf image as **healthy** or **diseased**.
Includes preprocessing, augmentation, training, testing, evaluation plots and prediction on unseen images.

**Student:** Dinesh Kumar N (Reg. No. 113024148019 / VH14114), CSE AIML - A, 3rd Year / 5th Sem
**College:** Vel Tech High Tech Dr. Rangarajan Dr. Sakunthala Engineering College (An Autonomous Institution)
**Subject:** Deep Learning Essentials | **Faculty:** Priyadarshini V

## Project structure
```
plant-leaf-cnn-project/
├── README.md
├── requirements.txt
├── dataset/                  # training images (see "Dataset")
├── src/
│   ├── leaf_disease_cnn.py   # train / test / predict
│   └── prepare_dataset.py    # builds the training set and held-out crops
├── notebooks/Leaf_Disease_CNN.ipynb
├── results/                  # training curves, confusion matrix, sample predictions
├── screenshots/              # predictions on unseen web images
└── unseen/                   # put new leaf images here for prediction
```

## Dataset
PlantVillage (https://github.com/spMohanty/PlantVillage-Dataset), `color` folder.
Any folder whose name contains `healthy` is labelled healthy; all others are diseased.

Final model (Model 2): 11 crops (Apple, Blueberry, Corn, Grape, Orange, Peach, Potato, Raspberry,
Soybean, Squash, Tomato), 32 classes, at most 300 images per healthy class and 150 per diseased class
(6,002 images). Cherry, Strawberry and Pepper (240 images) are held out and never used for training.

```bash
python src/prepare_dataset.py --src PlantVillage-Dataset/raw/color --out dataset --heldout heldout
```
(Use the path of your `color` folder. The selection is seeded, but the exact images can differ from the original run.)

## Run
```bash
pip install -r requirements.txt
python src/leaf_disease_cnn.py train --data_dir dataset --epochs 25
python src/leaf_disease_cnn.py predict --image_dir unseen/
```

## Method
- Resize to 128x128, scale pixels to [0,1]
- Stratified 70/15/15 train/validation/test split
- Augmentation (training only): flips, rotation, zoom, contrast
- CNN: 4 x [Conv3x3 -> BatchNorm -> MaxPool] -> GAP -> Dense128 -> Dropout0.5 -> Sigmoid (258,881 parameters)
- Adam (1e-3), binary cross-entropy, class weights, EarlyStopping, ReduceLROnPlateau

## Results (Model 2, final)
| Test | Result |
|---|---|
| Test set (901 images) | accuracy 97.00 %, loss 0.0611, ROC-AUC 0.9987 |
| Confusion matrix [[TN,FP],[FN,TP]] | [[380, 3], [24, 494]] |
| Healthy: precision / recall | 0.941 / 0.992 |
| Diseased: precision / recall | 0.994 / 0.954 |
| Held-out crops (Cherry, Strawberry, Pepper; 240 images) | 91.7 % (220 / 240) |
| Unseen web images (10) | 10 / 10 (hand-picked, indicative only) |

Baseline (Model 1, only 3 crops / 6 classes): 99.36 % test accuracy, but it labelled some healthy
leaves of unfamiliar plants as diseased (19 of 21 distinct web images correct), which motivated Model 2.

## Limitations
- About 5 % of diseased leaves are missed (12 % on held-out crops).
- Trained on 11 crops of lab-style PlantVillage photographs; field images are harder.
- Some crops contain only healthy or only diseased classes, so the model may partly learn the crop.
- Binary output only (no disease name). Web tests are small and hand-picked.

## References
Hughes & Salathé (2015), PlantVillage, arXiv:1511.08060 | Mohanty et al. (2016), Frontiers in Plant Science 7:1419
