# InsectClassifier

Install python requirements
	- sklearn
	- matplotlib
	- cv2
	- pandas
	- torch
	- torch_optimizer
	- imgaug
	- tqdm
	- PIL
	- numpy
	- glob

Download Data at: https://doi.org/10.5683/SP2/LMRVFN

Run python watershed.py to create cropped images. 

Make sure original image has interested arthropods with a white background similar to example images.

Make sure images are named: "group"_"subgroup"_"uniqueID" such as "Araneae_Unknown_2020_10_16_4334". Captialization matters.

args:
 - 	-- data_name: name of dataset. Important to set for custom projects
 - 	-- threshold: black pixel value segmentation threshold
 - 	-- div: reduced resolution
 - 	-- extension: saved image extension
 - 	-- density: calculate pixel counts per watershed sample
 - 	-- multiprocess: enable multiprocessing

Run python train.py to train model.

args:
 - 	-- data_name: name of dataset. Must be the same as used in watershed.py
 - 	-- arch: neural net used. Must be one of "densenet", "resnet", or "mobilenet"
 -	-- batch_size: number of images considered per batch
 -	-- fold: data segmentation. 1-10 valid options
 -	-- epochs: number of training epochs
 -	-- img_size: image size used for training/predictions


Run python predict.py to create report on testing set.

Predict against images in the "test_images" directory

args:
 -	-- data_name: name of dataset. Must be the same as used in train.py
 -	-- arch: neural net used. Must be one of "densenet", "resnet", or "mobilenet". Must be one already trained
 -	-- batch_size: number of images considered per batch
 -	-- model_name: name of saved model. Found in models/{data_name}/{fold}_{accuracy}_{epoch}.pt
 -	-- ensemble: use all in models/{data_name}/ trained to make predictions
 -	-- img_size: image size used for training/predictions. Must be the same as train.py

Resulting predictions will be saved in predictions/{data_name}/predictions_{date}_{time}
