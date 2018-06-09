#name :ZHIWEI CHEN
#date: 04/06/2018
#This is an application of AI on recognizing flowers, please run train.py first then run predict.py
#for train.py, feel free to adjust values(epochos,learning rate, hidden_layers) to improve model accuray
import torch 
import matplotlib.pyplot 
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import numpy as np

#get data
def get_data(data_dir): 

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    #resize valid and test
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.RandomRotation(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder for train,valid and test
    dataset ={
            'train_datasets' : datasets.ImageFolder(train_dir, transform = train_transforms),
            'valid_datasets' : datasets.ImageFolder(valid_dir, transform = train_transforms),
            'test_datasets' : datasets.ImageFolder(test_dir, transform = train_transforms)
    }
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    loaders = {
            'trainloaders' : torch.utils.data.DataLoader(dataset['train_datasets'], batch_size=64, shuffle=True),
            'validloaders' : torch.utils.data.DataLoader(dataset['valid_datasets'], batch_size = 32),
            'testloaders' : torch.utils.data.DataLoader(dataset['test_datasets'], batch_size = 32)
    }
    
    class_to_idx = dataset['train_datasets'].class_to_idx
    return loaders,class_to_idx


def set_model(arch,hidden_units,learning_rate):
    '''
    This is the function for set a model 
    Arguments: arch,hidden_units,learning_rate 
    Return: model,criterion,optimizer
    '''
    if arch =='vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained = True)
    else:
        print("Model architecture is not available.")

        #freeze the parameter
    for param in model.parameters():
        param.required_grad = False
       
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout',nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(hidden_units, 1000)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    return model,criterion,optimizer


def pretrain_model(epoch,use_gpu,trainloaders,validloaders,model,criterion,optimizer):
    '''
    This is a function for pretrained model to learn data
    Arguments: epoch,use_gpu,trainloaders,validloaders,model,criterion,optimizer
    Return: none
    print out Train loss, validation loss and validation accuracy
    '''
    epochs = epoch
    steps = 0
    running_loss = 0
    print_every = 50
    
    for e in range(epochs):
        
    # Model in training mode, dropout is on
        if use_gpu:
            model = model.cuda()
        model.train()
        
        for images, labels in iter(trainloaders):
            steps += 1
       
        # Wrap images and labels in Variables so we can calculate gradients
            inputs = Variable(images)
            targets = Variable(labels)
            optimizer.zero_grad()
        
        #in GPU mode
            if model == model.cuda():
                inputs, targets = inputs.cuda(), targets.cuda()
        
        #backward and forward    
            output = model.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.data[0]
        
            if steps % print_every == 0:
            # Model in inference mode, dropout is off
                model.eval()            
                accuracy = 0
                test_loss = 0
                for ii, (images, labels) in enumerate(validloaders):
                                                  
                # Set volatile to True so we don't save the history
                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)
                
              # move tensor to GPU mode
                    if model == model.cuda():
                        inputs, labels = inputs.cuda(), labels.cuda()
                
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]
                
                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            
                print(
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validloaders)),
                  "Validation Accuracy %: {:.3f}".format(100*accuracy/len(validloaders)))
            
                running_loss = 0
            
            # Make sure dropout is on for training
                model.train()
    
   
def set_checkpoint(model,optimizer,epochs,learning_rate,hidden_units,file_path,class_to_idx,arch):
    '''
    This a a function to create a checkpoint of model
    Arguments: model,optimizer,epochs,learning_rate,hidden_units,file_path,class_to_idx,arch
    Return:None
    print out save checkpoint in a given file path
    '''
    model.class_to_idx = class_to_idx
    checkpoint = {
                  'arch':arch,
                  'state_dict': model.state_dict(),
                  'train_datasets' : model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'learning_rate' : learning_rate,
                  'hidden_units':hidden_units
                 }
    torch.save(checkpoint, file_path)
    print("Save checkpoint: {}".format(file_path))            
            

def load_check_point(file_path):
    
    '''
    This is a function to laod a checkpoint
    Argument: file_path
    Return: model
    '''
    checkpoint = torch.load(file_path)
    model,criterion,optimizer = set_model(checkpoint['arch'],checkpoint['hidden_units'],checkpoint['learning_rate'])
    
    model.class_to_idx = checkpoint['train_datasets']
    #to get the model
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer']) 
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #get a sample from PIL
    sample_im = Image.open(image)
    
    #resize 256 
    size = 256, 256
    sample_im.resize(size)
    
    #crops 224x 224 in the center
    width, height = sample_im.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    sample_im = sample_im.crop((left, top, right, bottom))
    
    #color channel between 0 - 1 rate
    np_im = np.array(sample_im)/255
    
    #normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize_im = (np_im - mean)/std
    
    #return sample_im array that color channel as the first dimension
    return torch.from_numpy(normalize_im.transpose((2,0,1)))


def predict(image_path, model, topk, use_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    #dropout off
    model.eval()
    
    # pass image_path to process_image function
    image = process_image(image_path)
    image = Variable(image.unsqueeze_(0))
       
    # transfer image to tensor
    np_im = np.array(image)
    tensor_im = torch.from_numpy(np_im).float()

    #if model is in cuda
    if use_gpu:
         tensor_im =  tensor_im.cuda()
    
    #pass the tensor_im to the model
    output = model.forward(tensor_im)
    
    #get prob
    # Model's output is log-softmax, take exponential to get the probabilities
    ps = torch.exp(output).data
    pro = torch.topk(ps, topk)[0].tolist()[0]
    #print(pro)
    
    #get the associate index from pro on top 5 then pass to classes
    index = torch.topk(ps, topk)[1].tolist()[0]
    classes = []
    
    #print(index)
    #append to classes from associate index in  train_datasets.class_to_idx
    for i in index:
        for key,value in model.class_to_idx.items():
            if value == i:
                classes.append(key)
                       
    return pro, classes


def get_name(classes,cat_to_name):
    '''
    This is a function to printout associate probablity of a given image name 
    '''
    name=[]
    for elements in classes:
        name.append(cat_to_name[elements])
   
    return name
    
