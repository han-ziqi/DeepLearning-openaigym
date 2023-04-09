# COMP532-assigment2
## Reinforce Learning base on A2C algorithm for control LunarLander-v2
### Folder contents
This folder contains three .py files and a folder.

+ train.py is used to train the lander for a smooth landing.
+ test.py is used to test the training results.
+ config.py is used to adjust some of the important parameters, such as the number of loops.
+ The models folder is used to cache the data from previous tests.
          

### Before you start

Please make sure you already installed 

TensorFlow-1.13.1
python 3.6

And make sure your python version is not higher than 3.8 if you use anaconda

### How to use

If you want to see the previous training results, simply run test.py directly without making any changes to the other files. 

If you want to retrain the model, change the parameters in the config.py file, then run train.py by uncomment lines 157 to 159 in train.py. This process will take about 20 minutes. When you have finished running, run test.py to check the training results.