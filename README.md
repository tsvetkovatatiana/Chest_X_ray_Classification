# Chest_X_ray_Classification
This project implements a binary classification model to detect Pneumonia from chest X-ray images. The model uses a pretrained ResNet34 backbone and is trained and evaluated using stratified five-fold cross-validation. The dataset consists of approximately 15,000 chest X-ray images labeled as either Normal or Pneumonia.

### Data
The dataset used in this project was sourced from [Kaggle: Chest X-Ray Dataset by Muhammad Rehan](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)
. The dataset originally contained three classes: Normal, Pneumonia, and Tuberculosis. For this project, only Normal and Pneumonia images were retained.

Images were resized to match the input requirements of ResNet, normalized according to pretrained parameters, and augmented using horizontal flips and minor color adjustments. Oversampling and class weighting were applied to address class imbalance.

### Model
You can choose the model in args.py There are pretrained models: resnet18, resnet34, resnet50.

## Training and Evaluation

The model was trained with the following strategies:
- Stratified five-fold cross-validation was used to ensure balanced representation of classes in each fold.

- The training process used mixed precision training (AMP) to optimize performance on GPUs. The key training loop is structured as follows:

```python
with amp_context():
    outputs = model(images)
    loss = criterion(outputs, labels)

if use_amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()

del images, labels, outputs, loss
```
- Data loaders were optimized for CUDA with pin memory and prefetching:

```python
pin_memory = (device.type == "cuda")
prefetch_factor = 1
```
- Memory management was carefully handled to prevent GPU memory leaks:

```python
del train_loader, val_loader, train_dataset, val_dataset, model
gc.collect()
torch.cuda.empty_cache()
```

Evaluation metrics included balanced accuracy, F1 score (macro), recall, and ROC-AUC. These metrics provide a comprehensive assessment of model performance, accounting for class imbalance and clinical relevance.

## How to run
To run the project and utilize GPU acceleration with CUDA, follow the steps below:
1. Clone repository

```bash

git clone https://github.com/tsvetkovatatiana/Chest_X_ray_Classification.git
cd Chest_X-ray_Classification
```
2. Install Dependencies
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install required Python packages:
```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy
```
Notes on CUDA:

The above command installs PyTorch with CUDA 11.8 support. Adjust the CUDA version according to your GPU and driver compatibility.

Ensure that your system has an NVIDIA GPU with CUDA installed. You can verify CUDA availability in Python:
```python

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

3. Prepare the Dataset
Download the Chest X-Ray dataset from Kaggle: [Chest X-Ray Dataset by Muhammad Rehan](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset?resource=download)
- Retain only the Normal and Pneumonia classes.
- Combine the train and validation folders.
- Create CSV files containing image paths and labels. For example, metadata.csv for training/validation data and test.csv for evaluation.

4. Configure Training Parameters
Adjust hyperparameters in the training script or through command-line arguments.

<br>
P.S. 
This project was initially developed as part of my university coursework. I would like to sincerely thank my teacher for providing the idea for this project and for offering careful guidance and support whenever it was needed.