# A HEPMASS Classifier

This is a simple neural network tasked to predict whether a signature is associated with a real high energy collision of particles or just background noise. The neural network which I developed can achieve around 80% accuracy in classifying samples. For more information please refer to the original dataset [site](http://archive.ics.uci.edu/ml/datasets/HEPMASS).

Made with PyTorch, Pandas and Numpy. The specific main requirements are in the `requirements.txt` file.

### Usage

A new neural network can be trained simply by running `python main.py`. The parameters (e.g. learning rate and epochs) can be adjusted in code.
