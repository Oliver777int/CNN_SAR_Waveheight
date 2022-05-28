#Description of code

This script includes a convolutional neural network (CNN), a function that builds a dataset
and ways of training and evaluating the CNN. This CNN is meant to predict the significant
wave height from SAR images. (To acquire preprocessed SAR images, check out the linked repo below)

#The CNN
- The Net() class includes the structure of the CNN and a simple forward pass
- The variable _to_linear finds the flattened output of the last convolutional layer
- All layers use batchnorm, ReLu and the conv layers also use Max pool.
#Build Dataset Class
This class builds the dataset. The function reads sub images of large SAR images.
To acquire these sub images, please visit this repository: https://github.com/Alvingee/SEEX15-22-12.
 After having the preprocessed sub images saved in a directory in the format
".tif" you are ready to build the dataset.

Since reading tif files is very slow, the pixels of the sub images are instead saved
as numpy arrays. They are named "training_data_x.npy". The reason for not saving the entire
dataset in one numpy array is because it causes memory problems. There are also numpy arrays
called test_x and test_y which are used for evaluating the model. When training the model
every numpy array is being shown to the network 1 time per epoch.

#Parameters to chose
The program includes boolean and global variables that needs to be manually set.
###Variables
- n = pixel size of sub images (nxn)
- BATCH_SIZE, learning_rate and EPOCHS
- path = path to folder where the .tif files are stored (Class variable, line 56)
###Booleans
- Rebuild_data: Rebuilds the dataset. Only needs to be run 1 time.
- Load_model: Loads the previously saved model, will error if no model is saved.
- Save_model: Saves the current model.
- show_histogram: Shows a histogram of the wave heights in the dataset
- run_training: Set to True if you want to train the model
- fully_evaluate_model: Evaluates the entire model
- mini_prediction: Evaluate on a small part of validation set (or training set if train_prediction is True)
- iterate_numpy_arrays: Should always be True, unless you only want to fully evaluate the model, then set this boolean
to False
- boj_evaluation: Evaluates on buoys. (requires buoy data and output folder)