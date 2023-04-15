# s23_deepLearning_miniproject
This is the submission of the ResNet model required for the mini project in ECE-GY 7123 Deep Learning for Spring 2023. This submission has been made by Kaustubh Mishra and Jayvardhan Singh.

The following python packages are required:

- torch
- torchsummary
- numpy
- tqdm
- multiprocessing

Install them manually or use the following python command:  `! pip install torch torchsummary numpy tqdm multiprocessing`

## Overview

`project_model.py` : The python file where the actual core model used in the project is written.

`DatasetLoader.py` : The python file that contains the DataFetcher class to load the training and testing data. This file loads the ***CIFAR-10*** dataset and apply various data Augmentation techniques

`training_nb.ipynb` : This notebook is used for the training the model.

`testing_nb.ipynb` : This notebook is used for the testing the model. The testing accuracy is `92.9%`

`trainNormalizedParameters.npz` : This contains the mean and standard deviation values of the training dataset, which are used for normalization. Used in `DatasetFetcher` class.

`project1_model.pt` : Contains the trained weights of the model.

