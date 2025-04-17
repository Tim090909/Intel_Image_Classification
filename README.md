# Intel Image Classification

This project is part of a computer vision project where we classify landscape images.

## 📁 Dataset

We use the Intel Image Classification Dataset which includes images in the following categories:

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

## Підготовка до виконання проекту

First of all looked at the all classes and number of images in each training class
![Classes](images/classes.png)

Than, as you can see in diagram the number of images in all classes quite simillar so we do not need to trim anything 
![Classes diagram](images/classes_diagram.png)

Also, let's look at the images form each class
![Images from each class](images/images_from_classes.png)

## Побудова базової моделі нейронної мережі

![Incorrect classification after RestNet50](images/training_1.png)
![Incorrect classification after RestNet50](images/chart_1.png)
![Incorrect classification after RestNet50](images/c_m_1.png)
![Incorrect classification after RestNet50](images/incorrect_classification_1.png)

## Оптимізація гіперпараметрів моделі нейронної мережі

![Incorrect classification after RestNet50](images/manual_tuning.png)
![Incorrect classification after RestNet50](images/training_2.png)
![Incorrect classification after RestNet50](images/chart_2.png)
![Incorrect classification after RestNet50](images/c_m_2.png)
![Incorrect classification after RestNet50](images/incorrect_classification_2.png)

## Transfer learning

![Incorrect classification after RestNet50](images/restnet_res.png)
![Incorrect classification after RestNet50](images/c_m_3.png)

![Incorrect classification after RestNet50](images/incorrect_classification_3.png)


## ✅ To Run

Upload the dataset to Google Colab or extract locally, then run each notebook cell in order.

