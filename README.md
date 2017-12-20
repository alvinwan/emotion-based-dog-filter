# Adversarial Examples in Computer Vision: Building then Fooling an Emotion-Based Dog Filter

This repository includes all source code for the tutorial on DigitalOcean with the same title, including:
- A real-time filter that adds dog masks to faces on a live feed.
- A dog filter that responds to your emotions. (Couldn't find a pug mask, so I used a cat.) A generic dog for smiling "happy", a dalmation for frowning "sad", and a cat for dropped jaws "surprise".
- Utilities used for portions of the tutorial, such as plotting and advesarial example generation.
- Simple convolutional neural network written in PyTorch, with pretrained model.
- Ordinary least squares and ridge regression models using randomized features.

created by [Alvin Wan](http://alvinwan.com), December 2017

![step_8_emotion_dog_mask](https://user-images.githubusercontent.com/2068077/34196964-36383d58-e519-11e7-92dc-2d7c33ab29bd.gif)


# Getting Started

If you want to forego the tutorial and try this filter out:

1. [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

2. Start by installing [PyTorch](http://pytorch.org).

3. Install all Python dependencies.

```
pip install -r requirements.txt
```

4. To use the emotion dog filter, navigate into `src`:

```
cd src
```

5. Launch the final script.

```
python step_8_dog_emotion_mask.py
```

## Acknowledgements

These models are trained on a Face Emotion Recognition (FER) dataset curated by Pierre-Luc Carrier and Aaron Courville in 2013, as published on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).
