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
from torch import tensor
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

def data_loader(data_dir):
    ''' Function Performs the Transformations, Arranges Datasets, and Creates Dataloaders
        Function returns the train/validation/test Dataloders '''
    # Setup the Correct Directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define Training Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(224),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Define Testing/Validation Transforms
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Define Datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define Dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    # Return Values
    return trainloader, validloader, testloader, train_data.class_to_idx




def train():
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate

    # Proceed to Create the Label Mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Create the classifier
    if opt.arch == "vgg16":
        print("VGG16 Model Chosen")
        model = models.vgg16(pretrained=True)
        model_output_layer = 25088
    elif opt.arch == 'vgg19':
        print("VGG19 Model Chosen")
        model = models.vgg19(pretrained=True)
        model_output_layer = 25088
    elif opt.arch == "densenet121":
        print("DenseNet121 Chosen")
        model = models.densenet121(pretrained=True)
        model_output_layer = 1024
    elif opt.arch == "alexnet":
        print("AlexNet Chosen")
        model = models.alexnet(pretrained=True)
        model_output_layer = 9216
    elif opt.arch == 'resnet50':
        print("ResNet50 Chosen")
        model = models.resnet50(pretrained=True)
        model_output_layer = 1024
    else:
        print("Error. Non-Supported Architecture Specified.")
    # Now We Proceed to Freeze Parameters to Train
    for param in model.parameters():
        param.requires_grad == False

    # Classifier
    classifier = nn.Sequential(
        nn.Linear(model_output_layer, 512),
        nn.ReLU(),
        nn.Dropout(opt.dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(opt.dropout),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )

    # Make the Classifier of the Model Our Classifier
    model.classifier = classifier

    # Determine if GPU is available
    if opt.gpu == True:
        torch.cuda.empty_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=opt.learning_rate)

    # Send the Model To The Device
    # Create Loaders Necessary for Training
    trainloader, validloader, testloader, model.class_to_idx = data_loader(opt.dir)
    model.to(device)

    # Now We Begin Training
    steps = 0
    training_list = []
    validation_list = []

    # Go Through Epochs
    for e in tqdm(range(epochs), position=1):
        running_loss = 0
        ## Go Through All Training Images
        for inputs, labels in trainloader:
            # Necessary Steps
            steps += 1
            # Move Inputs and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear All Gradients
            optimizer.zero_grad()
            # Make Predictions
            outputs = model.forward(inputs)
            # Calculate Loss
            loss = criterion(outputs, labels)
            # Back Propogation
            loss.backward()
            # Proceed to the Next Step
            optimizer.step()
            # Update the Loss
            running_loss += loss.item()
        running_loss = running_loss / len(trainloader)
        training_list.append(running_loss)
        ## Test Every Epoch
        model.eval()
        validation_loss = 0
        accuracy = 0
        for inputs, labels in validloader:
            # Move Inputs and Labels to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Stop Gradients Since we are not back-propogating
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                # Calculate Accuracy
                ps = torch.exp(outputs).data
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()
        validation_loss = validation_loss / len(validloader)
        accuracy = accuracy / len(validloader)
        validation_list.append(validation_loss)
        print("Epoch: {}/{}... ".format(e + 1, epochs),
              "Loss: {:.4f}".format(running_loss),
              "Validation Loss {:.4f}".format(validation_loss),
              "Accuracy: {:.4f}".format(accuracy))


    # Save the Model
    model.to('cpu')
    torch.save({'architecture': opt.arch,
                'learning_rate': learning_rate,
                'dropout': opt.dropout,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                'checkpoint.pth')

    # Create the Argumer Parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help="Epochs For Training")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch Size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--arch', type=str, default='vgg16', help="Architectures: vgg16, vgg19, DenseNet121,densenet201,alexnet,resnet50")
    parser.add_argument('--dir', type=str, default="flowers", help="Directory Model Were Datasets are in")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout Rate For Classifier Model")
    parser.add_argument('--gpu', action='store_true', help="Enable GPU For Training if Available")
    opt = parser.parse_args()
    # Train Model Normally
    train()