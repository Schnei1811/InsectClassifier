import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
from glob import glob


def buildImageAspectRatio(X_path, img_size):
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

def createData(data_name, img_size, X_paths, Y=np.array([])):
    reset = True

    if not os.path.exists(f"metadata/{data_name}/Arrays_Batches"):
        os.makedirs(f"metadata/{data_name}/Arrays_Batches")

    if not os.path.exists(f"metadata/{data_name}/Arrays_Data"):
        os.makedirs(f"metadata/{data_name}/Arrays_Data")

    data_batch = 0
    for i, X_path in enumerate(tqdm(X_paths)):
        if reset == True:
            reset = False
            X = np.expand_dims(buildImageAspectRatio(X_path, img_size), axis=0)
        else:
            X = np.vstack((X, np.expand_dims(buildImageAspectRatio(X_path, img_size), axis=0)))
        if not i == 0 and i % 999 == 0:
            reset = True
            np.save(f"metadata/{data_name}/Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            if len(Y): np.save(f"metadata/{data_name}/Arrays_Batches/{data_name}_Labels_{data_batch}_{len(Y)}.npy", Y)
            data_batch += 1
        if i == len(X_paths) - 1:
            np.save(f"metadata/{data_name}/Arrays_Batches/{data_name}_Input_{data_batch}_{len(X)}.npy", X)
            if len(Y): np.save(f"metadata/{data_name}/Arrays_Batches/{data_name}_Labels_{data_batch}_{len(Y)}.npy", Y)
            data_batch += 1

    data_paths = []
    for batch in range(data_batch):
        data_paths.append(glob(f'metadata/{data_name}/Arrays_Batches/{data_name}_Input_{batch}_*')[0])

    for i, data_path in enumerate(tqdm(data_paths)):
        data = np.load(data_path)
        if len(Y): labels = np.load(data_path.replace("Input", "Labels"))

        if i == 0:
            X = data
        else:
            X = np.vstack((X, data))    

    np.save(f'metadata/{data_name}/Arrays_Data/{data_name}_Input_{len(X)}.npy', X)
    if len(Y): np.save(f'metadata/{data_name}/Arrays_Data/{data_name}_Labels_{len(Y)}.npy', Y)

    print("Data Created: ", X.shape)

    shutil.rmtree(f'metadata/{data_name}/Arrays_Batches/')
