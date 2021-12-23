import cv2
import os
import numpy as np
import argparse
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
import csv
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

def build_data(preds_list, metadata):

    img_dict = {}   
    img_dict["total"] = {}
    for v in metadata.values():
        img_dict["total"][v] = 0

    for img_pair in preds_list:
        img_name = img_pair[0].split("\\")[1].split(".")[0]
        pred = img_pair[1]

        if img_name not in img_dict: 
            img_dict[img_name] = {}
            for v in metadata.values():
                img_dict[img_name][v] = 0


        img_dict[img_name][pred] += 1
        img_dict["total"][pred] += 1
    
    return img_dict

def build_figures(img_dict, report_dir):

    for img_name, data in img_dict.items():
        sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}
        plt.title(img_name)
        plt.ylabel("Number")
        plt.xlabel("Functional Group")
        plt.xticks(rotation=90)
        plt.bar(sorted_data.keys(), sorted_data.values())
        plt.savefig(f"{report_dir}\\{img_name}_bar.png", bbox_inches="tight")

        plt.clf()

        pie_dict = {f"{k} - {v}": v for k, v in sorted_data.items()}
        plt.title(img_name)
        patches, texts = plt.pie(pie_dict.values())
        labels = [*pie_dict.keys()]
        plt.legend(patches[::-1], labels[::-1], loc=1, fontsize=8)
        plt.savefig(f"{report_dir}\\{img_name}_pie.png")
        plt.clf()


def main(preds_list, metadata, report_dir):

    img_dict = build_data(preds_list, metadata)
    build_figures(img_dict, report_dir)



        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="lifeplan015")
    parser.add_argument("--pred", type=str, default="pred_ensemble_2021-12-21_02-16-48")
    args = parser.parse_args()   

    pred_dir:str = f"predictions\\{args.data_name}\\{args.pred}"
    report_dir = f"reports\\{args.data_name}\\{args.pred}"
    metadata_path = f"metadata\\{args.data_name}\\{args.data_name}_num_to_class.json"


    with open(f"{pred_dir}.csv") as csvfile:
        predReader = csv.reader(csvfile, delimiter=",")
        preds_list = [x for x in predReader][1:]

    with open(metadata_path) as f:
        metadata = json.load(f)

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    main(preds_list, metadata, report_dir)