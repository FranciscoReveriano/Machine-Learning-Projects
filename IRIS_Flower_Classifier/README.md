# Simple Classifier

- Author: Francisco Reveriano
- Institution: Duke University 
- Lab: Applied Machine Laboratory 
- Date: April 13, 2020

## Details
This is a simple classifier that trains on top of a pre-trained network. 
The pre-trained network is trained on ImageNet. The structure is very flexible and allows for multiple changes. 

## Files 
- train.py 
- predict.py 
- Image Classifier Project (Jupyter Notebook)

### Train
Optional changes to the train function. 
- epochs
    - Number of Epochs To Train
- batch size 
    - Batch Size for Each Epoch
- learning-rate 
    - Learning Rate (Utilizing Adam Optimizer)
- architecture 
    - Pre-Train Architecture
- directory
    - Directory of the dataset
- dropout
    - Dropout for the Classifier
- gpu 
    - Whether to allow GPU training

### Predict
- image-path
    - Path of the Image we want to classify
- model-path
    - Checkpoint where the weight is saved
- json-path
    - Path of the JSON file holding the score dictionary
- gpu
    - Use GPU or not
-topk
    - How many of the top results to return
- display
    - Whether to display Matplotlib results