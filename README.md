# Adversarial Examples in Computer Vision: Building then Fooling an Emotion-Based Dog Filter

This repository includes a dog filter that responds to your emotions. (Really, it's an "animal" filter. I couldn't find a mask for pugs, so I used a cat instead.) Specifically, it applies a standard dog mask for smiling, dalmation for frowns, and a cat for a big gaping mouth denoting surprise. All other source code for the tutorial at DigitalOcean can be found here, including utilities used for portions of the tutorial.

created by [Alvin Wan](http://alvinwan.com), December 2017

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