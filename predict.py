import argparse
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import torch, torchvision
import torch.nn as nn
from torchvision import models, transforms
import json
import csv

# Number of classes in the dataset
img_size = 224

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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_images, X_paths):
        self.X_images = X_images
        self.X_paths = X_paths
    def __len__(self):
        return len(self.X_images)
    def __getitem__(self, idx):
        sample = self.X_images[idx]
        sample = sample.astype("float32") / 255.0
        sample = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(sample)
        return (sample, self.X_paths[idx])


def buildImageAspectRatio(X_path):
    img = cv2.imread(X_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resize_x = int(img.shape[1] * img_size / max(img.shape))
    resize_y = int(img.shape[0] * img_size / max(img.shape))

    push_x = (img_size - resize_x) // 2
    push_y = (img_size - resize_y) // 2

    resized_img = cv2.resize(img, (resize_x, resize_y))

    canvas = np.zeros((img_size, img_size, 3)).astype("uint8") + 255
    canvas[push_y:resized_img.shape[0] + push_y, push_x:resized_img.shape[1] + push_x, :] = resized_img
    return canvas

def createData(data_name, X_paths):

    if not os.path.exists("Arrays_Batches"):
        os.makedirs("Arrays_Batches")

    if not os.path.exists("Arrays_Data"):
        os.makedirs("Arrays_Data")

    reset = True

    data_batch = 0
    for i, X_path in enumerate(tqdm(X_paths)):
        if reset == True:
            reset = False
            X = np.expand_dims(buildImageAspectRatio(X_path), axis=0)
        else:
            X = np.vstack((X, np.expand_dims(buildImageAspectRatio(X_path), axis=0)))
        if not i == 0 and i % 999 == 0:
            reset = True
            np.save(f"Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            data_batch += 1
        if i == len(X_paths) - 1:
            np.save(f"Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            data_batch += 1

    data_paths = []
    for batch in range(data_batch):
        data_paths.append(glob(f'Arrays_Batches/{data_name}_Input_{batch}_*')[0])

    for i, data_path in enumerate(tqdm(data_paths)):
        data = np.load(data_path)

        if i == 0:
            X = data
        else:
            X = np.vstack((X, data))

    np.save(f'Arrays_Data/{data_name}_Input_{len(X)}.npy', X)

def test_model(model, dataloader, device, num_to_class, report_csv):

    model.eval()

    preds_array = np.array([])

    for inputs, paths in tqdm(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        preds_cpu = preds.cpu().numpy()

        preds_array = np.append(preds_array, preds_cpu)

        for i, pred in enumerate(preds_cpu):
            img_name = paths[i].split("/")[-1]            
            report_csv.append([img_name, num_to_class[pred]])

    csv_path = f"pred.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(report_csv)


def main(data_name, arch, model_name, batch_size):

    report_csv = [["file_path", "prediction (Order_Family)"]]

    with open(f"metadata/{data_name}_num_to_class.json") as f:
        num_to_class = json.load(f)
    num_to_class = {int(k):v for k,v in num_to_class.items()}
    num_classes = len(num_to_class)

    X_paths = glob("extracted/*")

    input_file_path = f"Arrays_Data/{data_name}_Input_{len(X_paths)}.npy"

    if not os.path.exists(input_file_path):
        createData(data_name, X_paths)
    X = np.load(input_file_path)

    image_dataset = CustomDataset(X, X_paths)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model_ft = initialize_model(arch, num_classes)
    # Detect if we have a GPU available
    # if torch.cuda.device_count() > 1:        
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    model_path = os.path.join("models", arch, model_name)
    model_ft.load_state_dict(torch.load(model_path))

    test_model(model_ft, dataloader, device, num_to_class, report_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Alus")
    parser.add_argument("--arch", default="mobilenet")          #densenet,  resnet, mobilenet
    parser.add_argument("--model_name", default="0_0.9765853658536585_450.pt")
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()




    main(args.data_name, args.arch, args.model_name, args.batch_size)