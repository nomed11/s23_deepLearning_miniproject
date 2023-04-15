import multiprocessing
import torchvision
import numpy as np
from torch.utils.data import DataLoader

class DatasetFetcher:

    def __init__(self, dataset="CIFAR10", batch_size=64):
        print("Initializing fetching %s dataset using torchvision"%(dataset))
        self.datasetObject = torchvision.datasets.__dict__.get(dataset, None)
        if self.datasetObject == None:
            raise Exception("Dataset %s not available in torchvision."%(dataset))
        self.batch_size = batch_size
        self.train_transformers = []
        self.test_transformers = []
        self.workersAvailable = min(multiprocessing.cpu_count(), 14)

    def addHorizontalFlipping(self):
        self.train_transformers.append(torchvision.transforms.RandomHorizontalFlip())

    def addVerticalFlipping(self):
        self.train_transformers.append(torchvision.transforms.RandomVerticalFlip())

    def addRandomCrop(self, size=32, padding=3):
        self.train_transformers.append(torchvision.transforms.RandomCrop(size=size, padding=padding))
        
    def addHistogramEqualization(self):
        self.train_transformers.append(torchvision.transforms.functional.equalize)
        self.test_transformers.append(torchvision.transforms.functional.equalize)

    def __addToTensor(self):
        self.train_transformers.append(torchvision.transforms.ToTensor())
        self.test_transformers.append(torchvision.transforms.ToTensor())
        
    def __loadTrainNormalizers(self):
        params = np.load("./trainNormalizedParameters.npz")
        return params['mean'], params['std']

    def addNormalizer(self):
        self.__addToTensor()
        trainingDataset = self.datasetObject(root="./data", train=True, download=True)
        trainData = trainingDataset.data/255.0
        mean = trainData.mean(axis=(0, 1, 2))
        std = trainData.std(axis=(0, 1, 2))
        np.savez("./trainNormalizedParameters", mean=mean, std=std)
        self.train_transformers.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.test_transformers.append(torchvision.transforms.Normalize(mean=mean, std=std))
        
    def addAutoAugmentation(self):
        self.train_transformers.append(torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10))

    def getLoaders(self):
        if len(self.train_transformers) == 0:
            self.__addToTensor()
        trainingDataset = self.datasetObject(root="./data", train=True, download=True, transform=torchvision.transforms.Compose(self.train_transformers))
        testingDataset = self.datasetObject(root="./data", train=False, download=True, transform=torchvision.transforms.Compose(self.test_transformers))
        trainLoader = DataLoader(trainingDataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workersAvailable)
        testLoader = DataLoader(testingDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workersAvailable)
        return trainLoader, testLoader
    
    def getTestLoader(self):
        mean, std = self.__loadTrainNormalizers()
        self.test_transformers.append(torchvision.transforms.ToTensor())
        self.test_transformers.append(torchvision.transforms.Normalize(mean=mean, std=std))
        testingDataset = self.datasetObject(root="./data", train=False, download=True, transform=torchvision.transforms.Compose(self.test_transformers))
        testLoader = DataLoader(testingDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workersAvailable)
        return testLoader

if __name__ == "__main__":
    df = DatasetFetcher(dataset="CIFAR10", batch_size=64)
    df.addHorizontalFlipping()
    df.addRandomCrop(size=32, padding=3)
    df.addHistogramEqualization()
    df.addNormalizer()
    trainLoader, testLoader = df.getLoaders()