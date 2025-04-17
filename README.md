# 🧠 Intel Image Classification

This project is part of a computer vision lab where we classify landscape images using PyTorch.

## 📁 Dataset

We use the [Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification) which includes images in the following categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The dataset is structured into:

```
dataset/
├── seg_train/
│   └── [class folders]
├── seg_test/
│   └── [class folders]
└── seg_pred/
    └── [unlabeled images]
```

## ⚙️ Environment Setup

```bash
pip install torch torchvision matplotlib seaborn
```

Or use the pre-installed environment in [Google Colab](https://colab.research.google.com/).

## 🚀 Project Features

- Dataset loading and exploration
- Image preprocessing (resize, normalization)
- Data augmentation (random flip, rotation, jitter)
- Custom dataset class for unlabeled prediction data
- Dataloaders for training, validation, testing
- Class distribution visualization
- Sample image previews with class names

## 📊 Training Setup

```python
batch_size = 32
image_size = (128, 128)
augmentation = True
```

## 📈 Visualizations

- Histogram of class distribution using Seaborn
- Preview of example images with labels

## ✅ To Run

Upload the dataset to Google Colab or extract locally, then run each notebook cell in order.

---

### 🧑‍💻 Author

**Your Name** - [yourgithub](https://github.com/yourgithub)
