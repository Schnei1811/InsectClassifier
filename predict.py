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
from datetime import datetime
from datetime import date

from utils_data_creation import createData

# Number of classes in the dataset

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


def test_model(model, dataloader, device, num_to_class, report_csv, data_name):

    global ensemble_dict
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
            if img_name not in ensemble_dict:
                ensemble_dict[img_name] = [pred]
            else:
                ensemble_dict[img_name].append(pred)

    if not os.path.exists(f"predictions\\{data_name}"):
        os.makedirs(f"predictions\\{data_name}")

    pred_path = f"predictions\\{data_name}\\pred_{date.today()}_{str(datetime.now().time()).replace(':','-').split('.')[0]}.csv"

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(report_csv)


def main(data_name, arch, batch_size, img_size, model_path, num_to_class):

    report_csv = [["file_path", "prediction (Order_Family)"]]

    num_to_json_path = f"metadata/{data_name}/{data_name}_num_to_class.json"

    if not os.path.exists(num_to_json_path):
        print(f"Need to Create Num to JSON file for data_name {data_name}. Run Train with new data")
        quit()

    num_classes = len(num_to_class)

    if not os.path.exists(f"test_images/{data_name}"):
        os.makedirs(f"test_images/{data_name}")

    X_paths = glob(f"test_images/{data_name}/*")
    if len(X_paths) == 0:
        print(f"No images found in test_images/{data_name}")
        quit()

    input_file_path = f"metadata/{data_name}/Arrays_Data/{data_name}_Input_{len(X_paths)}.npy"

    if not os.path.exists(input_file_path):
        createData(data_name, img_size, X_paths)
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

    model_ft.load_state_dict(torch.load(model_path))

    test_model(model_ft, dataloader, device, num_to_class, report_csv, data_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="alus")
    parser.add_argument("--arch", default="mobilenet")          #densenet,  resnet, mobilenet
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--model_name", default="0_0.125_0.pt")    
    parser.add_argument("--ensemble", default=False, type=bool)
    parser.add_argument("--img_size", default=244, type=int)
    args = parser.parse_args()


    global ensemble_dict
    ensemble_dict = {}

    model_dir = f"models/{args.data_name}/{args.arch}/*.pt"

    with open(f"metadata/{args.data_name}/{args.data_name}_num_to_class.json") as f:
        num_to_class = json.load(f)
    num_to_class = {int(k):v for k,v in num_to_class.items()}

    if args.ensemble == True:
        for model_path in glob(model_dir):
            main(args.data_name, args.arch, args.batch_size, args.img_size, model_path, num_to_class)

        ensemble_csv = [["file_path", "prediction (Order_Family)"]]
        
        for img_name in ensemble_dict:
            ensemble_csv.append([img_name, num_to_class[max(set(ensemble_dict[img_name]), key=ensemble_dict[img_name].count)]])

        ensemble_path = f"pred_ensemble.csv"

        with open(ensemble_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(ensemble_csv)

    else:
        model_path = os.path.join("models", args.data_name, args.arch, args.model_name)
        main(args.data_name, args.arch, args.batch_size, args.img_size, model_path, num_to_class)
