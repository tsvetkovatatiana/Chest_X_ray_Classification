# Chest X-ray Pneumonia Classification

Binary classification of chest X-ray images as **Normal** or **Pneumonia** using transfer learning with a pretrained ResNet backbone (ResNet18 / ResNet34 / ResNet50), trained with stratified 5-fold cross-validation and mixed-precision GPU training.

## Results

Best model: **ResNet34**, fold 4, 5 epochs, evaluated on the held-out test set (never used during training or model selection).

| Metric | Validation (fold 4, best epoch) | Test |
|---|---|---|
| Balanced accuracy | 0.917 | **0.909** |
| F1-score (macro) | 0.899 | **0.888** |
| Recall (pneumonia) | 0.995 | **0.996** |
| ROC-AUC | 0.971 | **0.974** |

<p align="center">
  <img width="420" alt="Confusion matrix on the test set" src="https://github.com/user-attachments/assets/3784376d-a4b8-4ea1-b173-fba849d7a115" />
  <br>
  <em>Confusion matrix on the held-out test set</em>
</p>

- Test performance is close to validation performance, so the model generalizes to unseen data.
- Training and validation loss decreased together (train 0.235 → 0.188, validation 0.216 → 0.190), with no sign of overfitting.
- Recall is very high: the model misses almost no pneumonia cases. Most remaining errors are normal images predicted as pneumonia.

### Training (fold 4)

<table>
  <tr>
    <td align="center" width="50%">
      <img alt="Training and validation loss, fold 4" src="https://github.com/user-attachments/assets/02b6c2bd-b451-4334-a2be-18ef8f7101b5" />
      <br><em>Training and validation loss</em>
    </td>
    <td align="center" width="50%">
      <img alt="Confusion matrix on validation, fold 4" src="https://github.com/user-attachments/assets/763eed81-9bcd-45ba-a38d-8b5cb54357ce" />
      <br><em>Confusion matrix on the validation fold</em>
    </td>
  </tr>
</table>

### Grad-CAM

Heatmaps of the image regions the model relied on (see [Model explainability](#model-explainability-grad-cam)). Click to expand.

<details>
  <summary><b>Grad-CAM: random test images and misclassified images</b></summary>
  <br>
  <table>
    <tr>
      <td align="center" width="50%"><b>Random test images</b></td>
      <td align="center" width="50%"><b>Misclassified images</b></td>
    </tr>
    <tr>
      <td align="center" valign="top">
        <img alt="Grad-CAM on random test images" src="https://github.com/user-attachments/assets/2fd3eaac-9aa9-4330-bba4-e3a572bda49f" />
      </td>
      <td align="center" valign="top">
        <img alt="Grad-CAM on misclassified test images" src="https://github.com/user-attachments/assets/5805d325-cd53-485c-8c33-b38d0b743bf6" />
      </td>
    </tr>
  </table>
</details>


## What changed from the first version

The first version of this project reached 0.91 balanced accuracy on validation but only **0.54 on the test set**: it predicted *pneumonia* for almost every image. Debugging it led to these fixes:

1. **Class imbalance was corrected twice.** A `WeightedRandomSampler` already balanced the classes 50/50, and `BCEWithLogitsLoss(pos_weight=3.0)` additionally made every pneumonia error count three times more. Since pneumonia is already the larger class, the model learned to predict it almost always. **Fix:** imbalance is now handled only by the sampler.
2. **ROC-AUC was computed on 0/1 predictions** instead of probabilities, which made it identical to balanced accuracy. **Fix:** ROC-AUC now uses predicted probabilities.
3. **Validation and test images were augmented** (random flips and brightness changes) because the dataset defaulted to training mode. **Fix:** validation and test use `train=False`.
4. **The learning-rate scheduler could never lower the learning rate**, because `min_lr` was equal to the learning rate. **Fix:** `-min_lr 1e-6`.
5. **Validation now runs under `torch.no_grad()`**, and unreadable image files are detected and skipped when the CSVs are built.

Result: test balanced accuracy improved from **0.54 to 0.90**.

## Data

Dataset: [Chest X-Ray Dataset by Muhammad Rehan (Kaggle)](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset).

The original dataset has three classes: Normal, Pneumonia and Tuberculosis. Only **Normal** and **Pneumonia** are used. The original `train` and `val` folders are combined and split into 5 stratified folds, and the original `test` folder is kept aside as an independent test set.

Preprocessing and augmentation:
- images are read in grayscale and converted to 3 channels for the pretrained ResNet
- resized to 224×224 and normalized with ImageNet mean and standard deviation
- training only: random horizontal flip and small brightness/contrast changes
- class imbalance: oversampling with `WeightedRandomSampler`

## Model and training

- **Backbone:** pretrained ResNet18, ResNet34 or ResNet50 (torchvision), with the final layer replaced by a single output for binary classification
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam (learning rate 1e-3, weight decay 1e-4)
- **Scheduler:** ReduceLROnPlateau on validation ROC-AUC (also supports cosine and step)
- **Mixed precision (AMP)** on CUDA GPUs for faster training and lower memory use
- **Checkpointing:** the best epoch per fold is saved based on validation balanced accuracy
- **Metrics:** balanced accuracy, macro F1-score, recall and ROC-AUC. These are chosen over plain accuracy because the classes are imbalanced, and in a medical setting a missed pneumonia case is the costliest error.

## Model explainability (Grad-CAM)

`gradcam.py` shows which image regions the model uses for its prediction. Heat on the lungs means the model relies on medically meaningful areas. Heat on corners, text markers or borders would point to shortcut learning (learning how the images were taken rather than the disease).

## Project structure

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Builds `metadata.csv`, `test.csv` and the 5 stratified fold CSVs; skips unreadable images |
| `dataset.py` | PyTorch `Dataset` that loads the X-ray images |
| `transforms.py` | Training and validation transforms |
| `model.py` | `TransferModel`: ResNet backbone with a binary output layer |
| `trainer.py` | Training and validation loop with AMP, checkpointing and plots |
| `main.py` | Runs 5-fold cross-validation |
| `evaluate.py` | Evaluates a saved model on the test set |
| `gradcam.py` | Grad-CAM heatmaps for model explainability |
| `utils.py` | Metrics, scheduler, checkpointing, plotting |
| `args.py` | Command-line arguments |

## How to run

### 1. Clone and install

```bash
git clone https://github.com/tsvetkovatatiana/Chest_X_ray_Classification.git
cd Chest_X_ray_Classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install PyTorch with CUDA support first (choose the CUDA version that matches your driver at [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Prepare the dataset

Download and unzip the dataset from Kaggle, then point `--data_root` to the folder that contains `train/`, `val/` and `test/`:

```bash
python prepare_dataset.py --data_root archive --out_dir data/CSVs
```

Rename `data/CSVs/test.csv` to `data/CSVs/test_data.csv` (the name `evaluate.py` expects).

### 3. Train (5-fold cross-validation)

```bash
python main.py -backbone resnet34 -epochs 5 -batch_size 32 -lr 1e-3 -min_lr 1e-6 -scheduler plateau -csv_dir data/CSVs -output_dir session
```

Best checkpoints, loss curves and confusion matrices are saved in `session/`. If you run out of GPU memory, use `-batch_size 16`.

### 4. Evaluate on the test set

```bash
python evaluate.py -backbone resnet34 -csv_dir data/CSVs -best_model_pth session/fold_1_best.pth
```

Results are saved in `evaluation_results/test_metrics.csv`.

### 5. Grad-CAM

```bash
python gradcam.py --model_path session/fold_1_best.pth --csv data/CSVs/test_data.csv
python gradcam.py --model_path session/fold_1_best.pth --csv data/CSVs/test_data.csv --mode errors
```

Heatmaps are saved in `gradcam_results/`.

## Possible improvements

- Tune the decision threshold on validation data instead of using 0.5, to reduce false positives while keeping recall high
- Average predictions from all 5 fold models (ensemble)

<br>
P.S.
This project was initially developed as part of my university coursework. I would like to sincerely thank my teacher for providing the idea for this project and for offering careful guidance and support whenever it was needed.
