# DeepLearning-Openaigym

This is a deep reinforcement learning project that I worked with a group of 5 people. We implemented [Openai Gym's Lunar Lander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/), which aims to teach the agent to land a spacecraft on the moon using techniques. The project utilizes **Python** as the primary programming language and several libraries such as **TensorFlow** and **NumPy**. 

My role was collaborate with another to designed a deep neural network using TensorFlow, with multiple layers of convolutional and dense layers to enable the agent to learn features and patterns from the environment. This project offers a great opportunity to gain hands-on experience in deep learning, reinforcement learning and develop an understanding of how machine learning algorithms can solve complex real-world problems.

In this project, we

- Compared the [Q-learning algorithm](https://en.wikipedia.org/wiki/Q-learning) with the AC algorithm 
- Introduced the [A2C algorithm](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- Implemented the A2C algorithm and finish deep learning training.

### Content
- train.py is used to train the lander for a smooth landing.
- test.py is used to test the training results.
- config.py is used to adjust some important parameters, such as the number of loops.
- The models folder is used to cache the data from previous tests.
          
### Before you start
Please make sure you already installed 

- TensorFlow-1.13.1
- python 3.6

And make sure your python version is not higher than 3.8 if you use anaconda

### How to use

If you want to see the previous training results, simply run test.py directly without making any changes to the other files. 

If you want to retrain the model, change the parameters in the config.py file, then run train.py by uncomment lines 157 to 159 in train.py. This process will take about 20 minutes. When you have finished running, run test.py to check the training results.

### demo
1. This is a demo to showing the result of training
![training demo](https://github.com/han-ziqi/DeepLearning-openaigym/raw/master/demo/Train%20result.jpeg "demo for training")

2.This is a demo for showing the result of test, the LunaLander can land smoothly and safe.
![finish demo](https://github.com/han-ziqi/DeepLearning-openaigym/raw/master/demo/demo.jpeg)

3.This is a gif to show the landing process.
**This gif is from [OpenAI gym](https://www.gymlibrary.dev), I use it to demonstrate it on behalf of my project**
![demo](https://github.com/han-ziqi/DeepLearning-openaigym/raw/master/demo/lunar_lander_continuous.gif)
