# A Retinal Diseases Classification Task :eyes: :capital_abcd:

## Overview
A retinal diseases classifier that classifies fundus retinal images into four categories of **Normal, Cataract, Diabetic Retinopathy, and Glaucoma**.

## Data set :file_folder:
The dataset consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images. These images are collected from various sorces like IDRiD, Oculur recognition, HRF etc. You can find the data set [here](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification).
## Model 
The lightweight CNN proposed model consists of three convolution layers with some BatchNormalization and Dropout layers used for regularization.

```python
class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
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
        )
        self.fc = nn.Sequential(
            nn.Linear(64*30*30, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, k)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*30*30)
        x = self.fc(x)
        
        return x
```
The model is trained with **90** epochs, and **98%** of training accuracy is gained.

## Evaluation :white_check_mark:
Train accuracy: 98%<br>
Validation accuracy: 95%<br>
Test accuracy:  96%

