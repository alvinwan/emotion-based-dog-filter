# [How To Apply Computer Vision to Build an Emotion-Based Dog Filter in Python 3](https://www.digitalocean.com/community/tutorials/how-to-apply-computer-vision-to-build-an-emotion-based-dog-filter-in-python-3)

**Want an in-person tutorial with step-by-step walkthroughs and explanations? See the corresponding AirBnb experience for both beginner and experienced coders alike, at ["Build a Dog Filter with Computer Vision"](http://abnb.me/GFEpWpfUlO)**

This repository includes all source code for the [tutorial on DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-apply-computer-vision-to-build-an-emotion-based-dog-filter-in-python-3) with the same title, including:
- A real-time filter that adds dog masks to faces on a live feed.
- A dog filter that responds to your emotions. (Couldn't find a pug mask, so I used a cat.) A generic dog for smiling "happy", a dalmation for frowning "sad", and a cat for dropped jaws "surprise".
- Utilities used for portions of the tutorial, such as plotting and advesarial example generation.
- Simple convolutional neural network written in [PyTorch](http://pytorch.org), with pretrained model.
- Ordinary least squares and ridge regression models using randomized features.

created by [Alvin Wan](http://alvinwan.com), December 2017

![step_8_emotion_dog_mask](https://user-images.githubusercontent.com/2068077/34196964-36383d58-e519-11e7-92dc-2d7c33ab29bd.gif)

# Getting Started

> You can setup the repository using Python or view the web demo at [dogfilter.alvinwan.com](https://dogfilter.alvinwan.com)

For complete step-by-step instructions, see the [tutorial on DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-apply-computer-vision-to-build-an-emotion-based-dog-filter-in-python-3). This codebase was developed and tested using `Python 3.6`. If you're familiar with Python, then see the below to skip the tutorial and get started quickly:

> (Optional) [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

1. Install all Python dependencies.

```
pip install -r requirements.txt
```

2. Navigate into `src`.

```
cd src
```

3. Launch the script for an emotion-based dog filter:

```
python step_8_dog_emotion_mask.py
```

# How it Works

See the below resources for explanations of related concepts:

- ["Understanding Least Squares"](http://alvinwan.com/understanding-least-squares/)
- ["Understanding Neural Networks"](http://alvinwan.com/understanding-neural-networks/)

## Acknowledgements

These models are trained on a Face Emotion Recognition (FER) dataset curated by Pierre-Luc Carrier and Aaron Courville in 2013, as published on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).
