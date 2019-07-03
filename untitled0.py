#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pytorch has two main packages :
# torch and torchvision 

# torch is the main package 
#torchvision package that contain model , datasets and transforms methods


# In[2]:


import torch
import torchvision


# In[3]:


from torch import nn 
import torch.nn.functional as F

from torch.utils.data import DataLoader 
from torchvision.datasets import FashionMNIST 
from torchvision.models import vgg16
from torchvision import transforms


# ## Formla to claclulate output of conv layer
# 
#     input is : n,w,h
#     output :n_new , w_new , h_new
#     n_new = out_channels
#     w_new = (w - kernal + 2 padding)/stride + 1 
#     h_new = (h - kernal + 2 padding)/stride + 1 

# In[5]:


batch = 20
train_data = FashionMNIST(r'C:\Users\elfakhrany\Documents\session2\data',train=True,transform=transforms.ToTensor(),download=False)
trainloader = DataLoader(train_data,batch_size=batch,shuffle=True)


# In[8]:


# let's get some Info About data
iterator = iter(trainloader)
images , labels = iterator.next()
print("Num of images in  training data is {}".format(len(train_data)))
print('Images shape is : {}'.format(images.shape))
print('Lables shape is : {}'.format(labels.shape))


# In[9]:


#check labels 
print(labels[0:10])


# In[10]:


# so what is Data classes ? 
# we can find it from Data repo : https://github.com/zalandoresearch/fashion-mnist
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[11]:


# let's plot some images 
import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[12]:



images = images.numpy()
print(images.shape)
images = np.transpose(images,(0,2,3,1)) #pytorch images has shape (channels , w , h) but plt need (w,h,channels)
print(images.shape)


# In[13]:



fig = plt.figure(figsize=(15,5))
for i in range(20):
    ax = fig.add_subplot(2,10,i+1)
    ax.imshow(np.squeeze(images[i]),cmap='gray')
    ax.set_title(classes[labels[i]])


# In[16]:


class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        # Here we build our Network arch
        # we know that input shpe is (1,28,28)
        self.conv1 = nn.Conv2d(1,16,3,padding=1) #shpae = (16,28,28)
        self.conv2 = nn.Conv2d(16,32,3,padding=1) #shpae = (32,28,28)
        self.pool1 = nn.MaxPool2d(2,2) # shape (32, 14, 14)
        self.conv3 = nn.Conv2d(32,64,3) #shpae = (64 ,12,12)
        self.fc = nn.Linear(12*12*64,10)
    
    def forward(self,x):
        # x is data passed to out network so we mush feed it forword to all layers 
        # self.conv() return linear output so we use F.relu -Activation function- to non-linearity 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.softmax(self.fc(x),dim=1)
        
        return x


net = Net()
print(net)


# ### Test Model before training 

# We just want to know how our model will do with data without ant training 
# so we know **Load** test data and feed it to out <bold >net</bold> and claculate it's accurecy 

# In[17]:


test_data = FashionMNIST('./data/',train=False,transform=transforms.ToTensor())
testloader = DataLoader(test_data,batch_size=batch,shuffle=True)


# In[21]:


# get one batch of test data to feed it to model
images , lables = iter(testloader).next()
images , labels = images , labels
outs = net(images)
pre = torch.argmax(outs ,dim=1)
true = (pre == labels).sum().numpy()
print("Accurecy for one batch is :{}".format(true/batch))
print("predected classes is : {}".format(pre))
print("true classes is : {}".format(labels))


# ## Finally we train our model

# Before we start train we should define our loss function and out optimizer and num of epoch 

# In[22]:


from torch import optim


# In[23]:


cost = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters())


# In[25]:


epoch = 1 # we should start with small num to test our model and change any thing then we set final epoch 
num_of_batches = len(train_data)//batch
#print(num_of_batches)
print("Training start...")
net.train()
for i in range(epoch):
    epoch_err = 0.0
    batches_err = 0.0
    for batch_i , data in enumerate(trainloader):
        images , labels = data
        images , labels = images , labels
        opt.zero_grad()
        outs = net(images)
        err = cost(outs,labels)
        err.backward()
        opt.step()
        batches_err+=err.item()
        epoch_err+=batches_err
        if(batch_i%1000 == 999): #print avg batch loss every 1000 batches
            print("Epoch {} , Batch {} , Avg loss {}".format(i+1,batch_i+1,batches_err/1000))

            
            batches_err=0.0
    
    print("--------\n Epoch {} Error is : {}\n--------".format(i+1,epoch_err/(num_of_batches)))
print("Training Finished.")


# In[ ]:



# initialize tensor and lists to monitor test loss and accuracy
test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))


# set the module to evaluation mode
net.eval()

for batch_i, data in enumerate(testloader):
    
    # get the input images and their corresponding labels
    inputs, labels = data
    inputs = inputs
    labels = labels
    # forward pass to get outputs
    outputs = net(inputs)

    # calculate the loss
    loss = cost(outputs, labels)
            
    # update average test loss 
    test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
    
    # get the predicted class from the maximum value in the output-list of class scores
    _, predicted = torch.max(outputs.data, 1)
    
    # compare predictions to true label
    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
    
    # calculate test accuracy for *each* object class
    # we get the scalar value of correct items for a class, by calling `correct[i].item()`
    for i in range(batch):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# ## Save Model

# In[ ]:



model_dir = 'saved_models/'
model_name = 'model_1.pt'

torch.save(net.state_dict(), model_dir+model_name)


# ## Load Model

# In[ ]:


# instantiate your Net
# this refers to your Net class defined above
net = Net()

# load the net parameters by name
# uncomment and write the name of a saved model
net.load_state_dict(torch.load('saved_models/model_1.pt'))

print(net)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





