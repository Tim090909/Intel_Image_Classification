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

First of all, I looked at all classes and the number of images in each training class.
 
![Classes](images/classes.png)

Then I created `DataLoaders` and loaded datasets with the transformers. The test dataset was loaded with simple transformations, and the training dataset with augmentation.

Then, as you can see in the diagram, the number of images in all classes is quite similar, so we do not need to trim anything.  

![Classes diagram](images/classes_diagram.png)

Also, let's look at the images from each class. 

![Images from each class](images/images_from_classes.png)

As the conclusion to this part, I can say that all classes are well balanced, and the train dataset compared to the val(test) dataset is in a 14 to 3 ratio, which is a good number.

## Побудова базової моделі нейронної мережі

Then in this part, I created a simple CNN with 3 convolutional layers, max pooling, 2 fully connected layers, Dropout, and ReLU.

```python
class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

After 30 epochs of training, I got these results:

![Training simple CNN](images/training_1.png)

This can be visualized in these charts. As you can see, the model is a bit overfitted: although we have quite good results on the training dataset (97% accuracy), the validation dataset stopped improving at around 87%, which means that our model is great at guessing on the training data but not as good on unseen images.  

![Simple CNN results training charts](images/chart_1.png)

From this confusion matrix, you can see that our model has some problems distinguishing between mountains and glaciers, also with buildings and streets, and a little with the sea and mountain/glacier. All of this is due to high similarity between the image pairs.  

![Confusion Matrix](images/c_m_1.png)

Below you can see the incorrectly guessed images by our network. As for me, only the first and the fifth images have some uncertainty, the rest should be correctly identified. 

![Incorrect classification simple CNN](images/incorrect_classification_1.png)

As you can see in this part, although the model is a little overfitted, we get quite good results (87%) from the start. And I will try to improve it in the next parts.

## Оптимізація гіперпараметрів моделі нейронної мережі

I created a function that sets one of 27 combinations of parameters such as convolutional layers, hidden neurons, and learning rates. Based on the results, I chose the parameters that gave the smallest loss and used them in the previous CNN model. I want to mention that before I replaced the parameters in the CNN model with these ones, I had slightly worse results (lower accuracy, etc.).  

![Manual tuning results](images/manual_tuning.png)

Then I used **Optuna** to find the best parameters for the model, and with these parameters I trained a new model for 30 epochs.  

![Training](images/training_2.png)

As a result, we still have overfitting, but the chart looks better — the parabolic line is more straight and there are no jumps. Also, the loss for the validation dataset is smaller.  

![Chart 2](images/chart_2.png)

However, the confusion matrix doesn't show any changes. 

![Confusion matrix 2](images/c_m_2.png)

Also, we can see some images that are not correctly classified. As for me, all except the fourth image are difficult to identify.  

![Incorrect classification](images/incorrect_classification_2.png)

So as a result from this part, I improved our model a little, found the optimal parameters for the first CNN, but we still have some problems like model overfitting, wrong image classification, and low validation accuracy (under 85%).

## Transfer learning

For this, I used the pretrained network **ResNet50**.

I used freezing and unfreezing of layers at each epoch. I trained it for 10 epochs and got the following result:  
![ResNet50 results](images/restnet_res.png)

So as you can see, we achieved 91% accuracy, which is quite a good result for this difficult dataset.

Additionally, you can see in the confusion matrix that we have better results for streets/buildings and sea/mountains/glaciers. Although the problem with mountains/glaciers persists.  
![Confusion matrix after ResNet50](images/c_m_3.png)

And finally, here are some images that were incorrectly classified. As for me, all of them except images 4 and 6 are really difficult to identify.

![Incorrect classification after RestNet50](images/incorrect_classification_3.png)


## 🔮 Steps for Further Improvement

1. Use more advanced data augmentation to reduce overfitting and improve generalization.
2. Train for more epochs using early stopping to avoid overfitting.
3. Try different pretrained models like EfficientNet, DenseNet, or ViT.
4. Increase the dataset size using external data or synthetic data generation.
5. Analyze misclassified images in detail to understand the model’s weaknesses better.
6.  Perform fine-grained hyperparameter tuning using advanced search methods.


## ✅ To Run

Upload the dataset to Google Colab or extract locally, then run each notebook cell in order.

