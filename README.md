# EyeSee: A Deep Learning Model for Retinal Disease Classification 👁️

## Overview
This project aims to develop a high-performing \& lightweight CNN model for retinal diseases, which can accurately categorize fundus retinal images into four classes: Normal, Cataract, Diabetic Retinopathy, and Glaucoma.
## Data set
The dataset consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images. These images are collected from various sorces like IDRiD, Oculur recognition, HRF etc. You can find the data set [here](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification).
<p class="row" float="left" align="middle">
  <img src="/images/normal.jpg" width="200" title="Normal"/>
  <img src="/images/cataract.jpg" width="200" title="Cataract"/> 
  <img src="/images/dr.jpeg" width="200" title="Diabetic Retinopathy"/>
  <img src="/images/glaucoma.jpg" width="200" title="Glaucoma"/>
</p>

## Model Architecture
The EyeSeeNet model is a convolutional neural network (CNN) that consists of four convolutional layers, each followed by a batch normalization layer and a ReLU activation function, and a max pooling layer to reduce the spatial dimensions of the feature maps. The resulting feature maps are flattened and passed through three fully connected layers, each followed by a ReLU activation function and a dropout layer to prevent overfitting.
```python
class EyeSeeNet(nn.Module):
    def __init__(self, num_class):
        super(EyeSeeNet, self).__init__()
        self.num_class = num_class
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*14*14, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_class)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*14*14)
        x = self.fc(x)
        return x
```

# Results
The model was trained for 90 epochs using stochastic gradient descent (SGD) with a learning rate of 0.001, a momentum of 0.9, and a batch size of 32. During training, the best performing weights were saved based on the validation loss, and these weights were used to initialize the model when making predictions on new images. The model achieved an accuracy of 94% on the test set, indicating its effectiveness in accurately classifying retinal images.


The model's performance was evaluated using precision, recall, and F1-score for each class, as well as the overall accuracy. The precision values for each class were above 0.9, indicating that when the model predicts a certain class, it is highly likely to be correct. The recall values were also quite high, ranging from 0.90 to 0.97, indicating that the model is effective in identifying positive instances for each class. The F1-score, which provides a balanced measure of precision and recall, was also high for each class.
## Confusion Matrix
Here is the confusion matrix of the test dataset;
<p class="row" float="left" align="middle">
  <img src="/images/confusion_matrix.png" title="confusion matrix"/>
</p>

# Contribution
If you are interested in contributing to the project, please follow these steps:
1. Fork the repository
2. Create a new branch for your changes
3. Commit your changes and open a pull request
