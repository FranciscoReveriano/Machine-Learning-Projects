'''
Name: Francisco Reveriano
Institution: Duke  University
Date: 04/12/2020
'''

import argparse
import numpy as np
import time
import torch
import json
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def setup_model(architecture, dropout):
    ''' ReCreate The Model Used From Previous Examples For Inference'''
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
        model_output_layer = 25088
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        model_output_layer = 25088
    elif architecture == "densenet121":
        model = models.densenet121(pretrained=True)
        model_output_layer = 1024
    elif architecture == "alexnet":
        model = models.alexnet(pretrained=True)
        model_output_layer = 9216
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=True)
        model_output_layer = 1024
    else:
        print("Error. Architecture not specified.")
    # Now We Proceed to Freeze Parameters to Train
    for param in model.parameters():
        param.requires_grad == False

    # Classifier
    classifier = nn.Sequential(
        nn.Linear(model_output_layer, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )

    # Make the Classifier of the Model Our Classifier
    model.classifier = classifier
    return model

def load_model(path):
    '''Function Loads the Model From the Saved PyTorch Model'''
    checkpoint = torch.load(path)
    architecture = checkpoint["architecture"]
    learning_rate = checkpoint['learning_rate']
    dropout = checkpoint['dropout']
    model = setup_model(architecture, dropout)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint["state_dict"])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    # Now We need to transform the image
    ## We Can Utilize the transformations defined in the test
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    img_tensor = test_transforms(img)
    return img_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Variables
    image_path = opt.image_path
    model = opt.model_path
    json_path = opt.json_path
    topk = opt.topk

    # Label Mapping
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    # Utilize GPU if GPU is available
    if opt.gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    ## Load Model
    model = load_model(model)
    ## Send Model to GPU
    model.to(device)
    ## Preprocess Image
    image = process_image(image_path)
    ## Need To Convert to Four Dimensional Input
    image = image.unsqueeze(0)
    image = image.float()
    ## Send to Model
    output = model.forward(image.to(device))
    ## Results
    results = F.softmax(output.data, dim=1)
    ## Now we call topk Function
    probabilities, predicted_class = results.topk(topk)
    ## Now Perform Sanity
    a = np.array(probabilities[0])
    b = [cat_to_name[str(index + 1)] for index in np.array(predicted_class[0])]

    # Print
    for i in range(len(a)):
        print("Class: '{}' with Probability: '{}'".format(b[i], a[i]))

    if opt.display == True:
        # Plot the Image
        imshow(process_image(image_path))
        plt.title(b[0])
        plt.axis('off')

        # Setup The plot
        fig2, ax = plt.subplots()
        ax.barh(np.arange(len(a)), a)
        ax.set_yticks(np.arange(len(a)))
        ax.set_yticklabels(b)
        plt.show()
    return

# Create the Argument Parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="flowers/test/7/image_07211.jpg", help="Tested Image Path")
    parser.add_argument("--model_path", type=str, default="checkpoint.pth", help="Pytorch Model Path")
    parser.add_argument("--json_path", type=str, default="cat_to_name.json", help="JSON File Path")
    parser.add_argument('--gpu', action='store_true', help="Enable GPU For Training if Available")
    parser.add_argument('--topk', type=int, default=5, help="Number of Closes Class To Show")
    parser.add_argument("--display", action='store_true', help="Display Plots")
    opt = parser.parse_args()
    # Train Model Normally
    predict()
