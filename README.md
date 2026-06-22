# CASMEII Micro-Expression Recognition (LOSO)

This project implements a micro-expression recognition training pipeline based on the CASMEII dataset using Leave-One-Subject-Out (LOSO) cross-validation.

## Project Structure

- `main.py`: Training entry point. It reads manifest files, builds LOSO datasets, and trains the model fold by fold for each subject.
- `train.py`: Training and evaluation logic, including the optimizer, learning-rate scheduler, backbone freezing, testing, and model saving.
- `model.py`: Model definitions and network architecture.
- `data_utils.py`: Data loading, manifest parsing, LOSO subject grouping, label mapping, and related utilities.
- `requirements.txt`: Python dependencies.
- `data/`: Default data directory containing datasets such as `CASMEII`, `CASME`, and `SMIC`.
- `saved_models/`: Default directory for trained models and label-mapping files.
- `magnification_model/`: Adaptive magnification module.

## Environment Setup

Python 3.8 is recommended. Install the dependencies from the project root directory:

```bash
pip install -r requirements.txt
```

Main dependencies:

numpy==1.21.5
Pillow==8.4.0
scikit-learn==1.0.2
scipy==1.7.3
matplotlib==3.5.1
pandas==1.3.5
tqdm==4.62.3
tensorboard==2.7.0
opencv-python==4.5.5.64
PyYAML==6.0
torch==1.10.1
torchvision==0.11.2

To use a GPU, make sure that your CUDA and PyTorch versions are compatible. The code automatically selects `cuda:0` when available and falls back to the CPU otherwise.

## Training Command

Run training with the default configuration:

```bash
python main.py
```

## Training Details

- This project uses LOSO validation. In each fold, one subject is reserved as the test set, while all remaining subjects are used for training.
- By default, the ResNet18 backbone is frozen for the first `10` epochs. The backbone and classification head are then trained together.
- The best model is saved according to the UAR (Unweighted Average Recall) on the test set.
- After training on each dataset, the model weights and label-mapping files are saved in `saved_models/`, for example:
  - `V1.0.0_CASMEII_sub01_5cls.pth`
  - `CASMEII_label_map.json`

## Command-Line Arguments

- `--manifest`: List of manifest file paths.
- `--datasets`: List of dataset names to train on.
- `--frame_window`: Frame-window size.
- `--num_epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.
- `--min_lr`: Minimum learning rate.
- `--save_dir`: Directory in which models are saved.
- `--version`: Version prefix used for saved model files.
- `--freeze_backbone_epochs`: Number of epochs for which the backbone is frozen.
- `--backbone_lr_scale`: Learning-rate scale applied to the backbone.
- `--no_pretrained`: Disable loading ImageNet-pretrained weights.
- `--use_cudnn`: Enable cuDNN acceleration.

## Supported Datasets and Classification Protocols

The current implementation supports the following dataset protocols:

- `CASMEII`: 5 classes (`happiness`, `disgust`, `surprise`, `repression`, and `others`).
- `CASME`: 4 classes (`disgust`, `surprise`, `repression`, and `tense`).
- `SMIC`: 3 classes (`positive`, `negative`, and `surprise`).
