# Extra Assignment - Image Creation with Neural Networks
The goal of this assignment was to create an "art" piece using AI.
Main engine used in this exercise is [Tensorflow](https://www.tensorflow.org/).
 
The solution was to be documented using **Docstring**.

## Setup
**Important**

The solution uses **Tensorflow v.1.15** that requires python to be of versions **between 3.5 to 3.7**.
The following solution was built using **python 3.7.0**.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all of the dependencies:
```bash
pip install -r requirements.txt
```
to install all of the required packages for all exercises project


## About the program
This program changes the supplied photo according to the supplied style, using neural networks.

This program takes two main inputs:
- content.jpg
- style.jpg

**Content** is a photo that we want to apply the **Style** to.

**Style** is the style of an artist (i.e. a painting by Picasso or Rembrandt) you want to apply to the **Content**.

An output of the program is the intial photo (content.jpg) with an applied "filter" that will (hopefully) be similar in style to the photo in style.jpg. 

## How to run
In order to run the program you just need to do this:
```bash
cd into/project/path
python3 neural_style.py
```

**IMPORTANT**

Since this project uses pretrained network this file [imagenet-vgg-verydeep-19.mat](https://www.kaggle.com/teksab/imagenetvggverydeep19mat) needs to be present in the root of the repository.

## Examples

![Imgur](https://i.imgur.com/Kho7H1Y.jpg)

## Useful Links

### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)

[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)

### scikit-learn
[scikit-learn](https://scikit-learn.org/stable/)

[scikit-learn svm](https://scikit-learn.org/stable/modules/svm.html)

### Tensorflow
[Tensorflow](https://www.tensorflow.org/)