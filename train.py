from __future__ import print_function
from __future__ import division
import torch, torchvision
import torchvision.models as models
import os
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image, ImageDraw
from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import cv2
import random
import time
import math
from glob import glob
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import copy
import sys
import argparse
import csv
import torch_optimizer as torch_optimizer
from shutil import copyfile
import json
from bisect import bisect_left

from utils_data_creation import createData

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

TRAIN = 0
TEST = 1

# Set random seed
np.random.seed(0)

# Set number of folds
num_folds = 10


def set_parameter_requires_grad(model):
    modify_pretrained_params = True
    for param in model.parameters():
        param.requires_grad = modify_pretrained_params

def weights_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data)

class GlobalAvgPool2d(nn.Module):
    def forward (self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

def initialize_model(arch, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if arch == "resnet":
        """ Resnet101 """
        model_ft = models.resnet101()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif arch == "mobilenet":
        """ Mobilenet """
        model_ft = models.mobilenet_v2()
        num_ftrs = model_ft.classifier[1].in_features

    elif arch == "densenet":
        """ Densenet """
        model_ft = models.densenet201() #DenseNet201
        num_ftrs = model_ft.classifier.in_features
    else:
        print(f"Unknown model name {arch}. Choose from resnet, mobilenet, or densenet")
        quit()


    model_ft.classifier =  nn.Sequential(
                GlobalAvgPool2d(), #Equivalent to GlobalAvgPooling in Keras
#                nn.Linear(1920, 1024),
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes))

    return model_ft

# Initialize the model for this run# Print the model we just instantiated
#print(model_ft)

def f1_loss(y_pred, y_true, is_training=False):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def plot(metric, train, val, epoch, fold, data_name, arch):

    max_y = max(train + val)

    #plt.title(f"{round(max(val), 2)} - {metric} - {data_name} - {xp_description} - {model_name} - {fold}-fold")
    plt.title(f"{metric} - {data_name} - {arch} - {fold}-fold")
    plt.plot(range(epoch + 1), train, label="train")
    plt.plot(range(epoch + 1), val, label="val")
    plt.axis([0, len(train)-1, 0, max_y + 0.1])
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.savefig(f"models/{data_name}/{arch}/plot_{metric}_{fold}.jpg", bbox_inches="tight")
    plt.clf()


def train_model(model, dataloaders_dict, device, criterion, optimizer, fold, scheduler, data_name, epochs, arch):

    # Initialize training
    since = time.time()
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over number of epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Set model to training mode
                model.train()  
            else:
                # Set model to evaluate mode
                model.eval()   

            # Initialze epoch metrics
            running_loss, running_f1, running_corrects = 0.0, 0.0, 0

            # Iterate over data.
            # Get inputs, labels and path from Custom DataLoader
            for inputs, labels, path in tqdm(dataloaders_dict[phase]):

                # Send to GPU (if using GPU)
                inputs, labels = inputs.to(device), labels.to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs
                    outputs = model(inputs)

                    # Get model loss
                    loss = criterion(outputs, labels)

                    # Get model predictions from soft max
                    _, preds = torch.max(outputs, 1)

                    # Optimize if Train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update Running Statistics During Epoch

                # Loss
                running_loss += loss.item() * inputs.size(0)

                # F1
                running_f1 += f1_loss(preds, labels.data)

                # Accuracy
                running_corrects += torch.sum(preds == labels.data)

            # Summarize Epoch Statistics

            # Loss
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)

            # F1
            epoch_f1 = running_f1 / len(dataloaders_dict[phase].dataset)

            # Accuracy
            epoch_acc = (running_corrects.double() / len(dataloaders_dict[phase].dataset)).cpu()

            print('\n{} Loss: {:.4f} F1 {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_f1, epoch_acc))

            # If Train, Update Training Summary Statistics
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            # If Val, Update Validation Summary Statistics
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

            # If Best Performing model on validation set
            if phase == 'val' and epoch_acc > best_acc:

                # Set new best accuracy 
                best_acc = epoch_acc

                # Copy model. Because PyTorch
                best_model_wts = copy.deepcopy(model.state_dict())

                # Remove previous best model. Saves space to keep only the best model
                if "prev_best_name" in locals():
                    os.remove(prev_best_name)

                # Save best model
                fname = f"models/{data_name}/{arch}/{fold}_{(best_acc * 10**5).round()/(10**5)}_{epoch}.pt"
                torch.save(best_model_wts, fname)

                # Save name of save model as previous best for future iterative deletion
                prev_best_name = fname
                print(f'model succesfully saved with val_acc = {best_acc}')

        plot("loss", train_loss_history, val_loss_history, epoch, fold, data_name, arch)
        plot("acc", train_acc_history, val_acc_history, epoch, fold, data_name, arch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best train Acc: {:4f}'.format(train_acc_history[-1]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_report_csv(data_name, xp_description, model_name, fold):
    global mixed_dict
    global report_csv

    classifications = list(num_to_class.values())

    for sample in report_csv[1:]:
        photo_id = "_".join(sample[0].split("_")[:5])

        if photo_id not in mixed_dict:
            mixed_dict[photo_id] = {k:0 for k in classifications}
        mixed_dict[photo_id][sample[2]] += 1

    sorted_mixed_dict = dict(sorted(mixed_dict.items()))
    mixed_pred_csv = []
    headers = ["Photo_ID"]
    classifications = list(num_to_class.values())
    headers += classifications
    mixed_pred_csv.append(headers)
    for photo_id, preds in sorted_mixed_dict.items():
        preds_list = list(preds.values())
        row = [photo_id] + preds_list
        mixed_pred_csv.append(row)

    csv_path = f"Models/{data_name}/{xp_description}/{model_name}/pred_mixed_{fold}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mixed_pred_csv)

    mixed_dict = {}


def build_confusion_matrix(labels_array, preds_array, labels, fold, epoch_acc, data_name, xp_description, model_name):

    labels_class_array = np.array([num_to_class[int(i)] for i in labels_array])
    preds_class_array = np.array([num_to_class[int(i)] for i in preds_array])

    conf_mtrx = confusion_matrix(labels_class_array, preds_class_array, labels)

    conf_mtrx_norm = conf_mtrx / conf_mtrx.max(axis=0)
    conf_mtrx_norm = conf_mtrx / conf_mtrx.max(axis=1)

    cmap = colors.ListedColormap(['#ffffff', '#e8ffff', '#d9f1ff', '#bfe6ff', '#8cd3ff', 
                                    '#59bfff', '#26abff', '#0da2ff', '#009dff', '#0055ff'])

    bounds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plot = plt.imshow(conf_mtrx_norm, interpolation='nearest', cmap = cmap, norm=norm)

    plt.xticks([i for i in range(len(labels))], labels)
    plt.xticks(rotation=80)
    plt.yticks([i for i in range(len(labels))], labels)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediciton")
    plt.title(f"Confusion Matrix - {fold} - {round(epoch_acc.item(), 4)}acc")


    for i in range(conf_mtrx.shape[0]):
        for k in range(conf_mtrx.shape[1]):
            plt.text(k, i, str(conf_mtrx[i][k]), horizontalalignment='center', verticalalignment='center')

    plt.gcf().subplots_adjust(bottom=0.3)
    plt.savefig(f"Models/{data_name}/{xp_description}/{model_name}/plot_confusion_matrix_{fold}.jpg")
    plt.clf()


def split_data(X, n:int = num_folds):
    print('\nSplitting data')
    kf = KFold(n_splits=n, shuffle=False)
    kf.get_n_splits(X)
    kfolds = []
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "VAL:", val_index)
        kfolds.append((train_index, test_index))
    return kfolds


def yield_fold(X, Y, X_paths, train_val_or_test, chosen_fold, kfolds):
    #print("TRAIN LABELS", len(Y[kfolds[chosen_fold][TRAIN]]))
    #print("TEST LABELS", len(Y[kfolds[chosen_fold][VAL]]))

    val_index = int(len(X[kfolds[chosen_fold][TRAIN]]) * 0.9)

    if train_val_or_test == "train":
        train_X = X[kfolds[chosen_fold][TRAIN]][:val_index]
        train_Y = Y[kfolds[chosen_fold][TRAIN]][:val_index]
        train_paths = X_paths[kfolds[chosen_fold][TRAIN]][:val_index]
        return (train_X, train_Y, train_paths)
    elif train_val_or_test == "val":
        val_X = X[kfolds[chosen_fold][TRAIN]][:-val_index]
        val_Y = Y[kfolds[chosen_fold][TRAIN]][:-val_index]
        val_paths = X_paths[kfolds[chosen_fold][TRAIN]][:-val_index]
        return (val_X, val_Y, val_paths)
    else:
        test_X = X[kfolds[chosen_fold][TEST]]
        test_Y = Y[kfolds[chosen_fold][TEST]]
        test_paths = X_paths[kfolds[chosen_fold][TEST]]
        return (test_X, test_Y, test_paths)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_images, targets, X_paths, img_aug, transform):
        self.X_images = X_images
        self.target = targets
        self.img_aug = img_aug
        self.transform = transform
        self.X_paths = X_paths
    def __len__(self):
        return len(self.X_images)
    def __getitem__(self, idx):
        sample = self.X_images[idx]
        sample = self.img_aug.augment_image(sample).copy()
        sample = sample.astype("float32") / 255.0
        sample = self.transform(sample)
        #print("TRANSFORM", sample)
        return (sample, self.target[idx], self.X_paths[idx])


img_aug = {
    'train': iaa.Sequential([
                iaa.Fliplr(p=0.5),
                iaa.SomeOf((0, 1), [
                    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Affine: Scale/zoom,
                           translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate/move
                           rotate=(-90, 90), shear=(-4, 4)),  # Rotate and Shear
                    iaa.PiecewiseAffine(scale=(0, 0.05)),  # Distort Image similar water droplet  1.76
                    ]),
                iaa.SomeOf((0, 2),[
                    iaa.Add((-3, 3)),  # Overall Brightness
                    iaa.Multiply((0.95, 1.05), per_channel=0.2),  # Brightness multiplier per channel
                    iaa.Sharpen(alpha=(0.1, 0.75), lightness=(0.85, 1.15)),  # Sharpness
                    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',  # Random HSV increase
                                       children=iaa.WithChannels(0, iaa.Add((-20, 20)))),
                    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                                       children=iaa.WithChannels(1, iaa.Add((-20, 20)))),
                    iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                                       children=iaa.WithChannels(2, iaa.Add((-20, 20)))),
                    iaa.AddElementwise((-5, 5)),  # Per pixel addition
                    iaa.CoarseDropout((0.0, 0.015), size_percent=(0.02, 0.25)),  # Add large black squares
                    iaa.GaussianBlur(sigma=(0.1, 0.75)),  # GaussianBlur
                    iaa.Grayscale(alpha=(0.1, 1.0)),  # Random Grayscale conversion
                    iaa.Dropout(p=(0, 0.08), per_channel=0.2),  # Add small black squares
                    iaa.AdditiveGaussianNoise(scale=(0.0, 0.03 * 255), per_channel=0.5), # Add Gaussian per pixel noise
                    iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.15),  # Distort image by rearranging pixels
                    #iaa.weather.Clouds(),
                    iaa.weather.Fog(),
                    #iaa.weather.Snowflakes()

                ])

            ]),
    'val': iaa.Sequential([
                iaa.Fliplr(p=0)
            ]),
    'test': iaa.Sequential([
                iaa.Fliplr(p=0)
            ])
}

data_transforms ={
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}


def createCSV(csv_path, data_name):

    if not os.path.exists("CSVs"):
        os.makedirs("CSVs")

    csv_lst = []
    csv_lst.append(["filepath", "species"])

    img_lst = glob(f"extracted/{data_name}/*")

    if (len(img_lst)) == 0:
        print(f"NO IMAGES FOUND IN extracted/{data_name} FOLDER. Place images there or use watershed.py")

    for file_path in tqdm(img_lst):
        print(file_path)
        species = "_".join(file_path.split("/")[-1].split("\\")[-1].split("_")[:2])
        print(species)
        csv_lst.append([file_path, species])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_lst)


def main(data_name, arch, batch_size, fold, epochs, img_size):

    if not os.path.exists(f"models/{data_name}/{arch}"):
        os.makedirs(f"models/{data_name}/{arch}")

    if not os.path.exists(f"metadata/{data_name}"):
        os.makedirs(f"metadata/{data_name}")

    # Create Input CSV
    csv_path = os.path.join("CSVs", data_name) + '.csv'

    createCSV(csv_path, data_name)

    # Read data from input csv
    data = pd.read_csv(csv_path)
    num_to_class = {idx : species for idx, species in enumerate(data['species'].unique())}
    class_to_num = {species : idx for idx, species in enumerate(data['species'].unique())}
    num_classes = len(num_to_class)

    if num_classes == 0:
        print("Num Classes is 0. Check CSV creation")

    # Save num_classes for testing
    with open(f"metadata/{data_name}/{data_name}_num_to_class.json", "w") as fp:
        json.dump(num_to_class, fp)

    print(data)
    print("Num Classes: ", num_classes)

    # Create & Load data
    X_paths = np.array(data['filepath'])
    Y = np.array([class_to_num[each] for each in data['species']])

    input_file_path = f"metadata/{data_name}/Arrays_Data/{data_name}_Input_{len(X_paths)}.npy"
    label_file_path = f"metadata/{data_name}/Arrays_Data/{data_name}_Labels_{len(X_paths)}.npy"

    if not os.path.exists(input_file_path) or not os.path.exists(label_file_path):
        createData(data_name, img_size, X_paths, Y)
    X = np.load(input_file_path)
    Y = np.load(label_file_path)

    # Shuffle data
    permutation = np.random.permutation(len(Y))
    X = X[permutation]
    Y = Y[permutation]
    X_paths = X_paths[permutation]

    # Get data based on Fold
    kfolds = split_data(X)

    # Create train, val, and test image datasets
    image_datasets = {phase: CustomDataset(*yield_fold(X, Y, X_paths, phase, fold, kfolds), img_aug[phase], data_transforms[phase]) for phase in ['train', 'val', 'test']}

    # Initialize model
    model = initialize_model(arch, num_classes)

    # Detect if we have a GPU available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)


    # Build train and val pyTorch dataloaders
    dataloaders_dict = {phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4) for phase in ['train', 'val']}

    # Set Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr =0.000001, verbose=True)

    # Assign weights per class during training. Helps with overfitting to common classes
    unique, counts = np.unique(Y, return_counts=True)
    class_weights = [(max(counts)/ count)**0.5 for count in counts]

    # Setup the loss function
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Train and evaluate model
    model_ft = train_model(model, dataloaders_dict, device, criterion, optimizer, fold, scheduler, data_name, epochs, arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data_name", default="alus")
    parser.add_argument("--arch", default="mobilenet")          #densenet,  resnet, mobilenet
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--fold", default=0, type=int)           #train or test
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--img_size", default=244, type=int)
    args = parser.parse_args()

    main(args.data_name, args.arch, args.batch_size, args.fold, args.epochs, args.img_size)