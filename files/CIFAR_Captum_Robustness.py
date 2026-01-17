#!/usr/bin/env python
# coding: utf-8

# # Captum Robustness with Image Classification
# 
# 
# This tutorial demonstrates how to apply robustness tooling in captum.robust including adversarial attacks (FGSM and PGD) as well as robustness metrics to measure attack effectiveness and generate counterfactual examples.
# 
# In this tutorial, we use a simple image classification model trained on the CIFAR-10 dataset. Be sure to install the torchvision and matplotlib packages before you start.

# In[2]:


import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from captum.robust import FGSM
from captum.robust import PGD

# The cell below loads the test data, transforms the data to a tensor, defines necessary normalization and defines the label classes. For the purpose of this tutorial, we will only need the test dataset, because we will use a pretrained model in our `models` folder.

# In[3]:


transform_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
unnormalize = lambda x: 0.5*x + 0.5

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_tensor)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# The classification model under attack is defined below. It is the same as the CIFAR model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py.

# In[4]:


import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
softmax = nn.Softmax(dim=1)


# We define a few helper methods to obtain predictions for a given image and visualize image and results.

# In[5]:


def image_show(img, pred):
    npimg = img.squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred)
    plt.show()
    
def get_prediction(model, input, normalize_im=False):
    if normalize_im:
        input = normalize(input)
    output = model(input)
    _, pred = torch.max(output, dim=1)
    return classes[pred], softmax(output)[:,pred]

# To save training time, we will load the pretrained weights from `models/cifar_torchvision.pt` instead of training the model from scratch. You may also train your own CIFAR model.

# In[6]:


net.load_state_dict(torch.load('models/cifar_torchvision.pt'))

# Now we are ready for the central part of this tutorial - generating adversarial examples with FGSM and PGD from Captum and then applying robustness metrics to better understand model vulnerabilities and the decision boundary. 
# 
# We will pick one image from the test set to use as an example and generate a perturbed image (adversarial example) that is very similar to the original image but causes the model to make a false prediction.

# In[7]:


image, label = testset[11]
image = image.unsqueeze(0)

image.requires_grad = True
# Get original prediction
pred, score  = get_prediction(net, image, normalize_im=True)

# We first utilize the Fast Gradient Sign Method (FGSM) to construct an adversarial example. FGSM utilizes the sign of the gradient to perturb the input.
# 
# We now construct the FGSM object, providing the desired lower and upper bound of perturbed inputs. In this case, since we are working with normalized inputs, the values should be in the range -1 to 1. We then call perturb on the FGSM object, providing epsilon (magnitude of the perturbation) as well as the label.

# In[8]:


# Construct FGSM attacker
fgsm = FGSM(net, lower_bound=-1, upper_bound=1)
perturbed_image_fgsm = fgsm.perturb(normalize(image), epsilon=0.16, target=label) 
new_pred_fgsm, score_fgsm = get_prediction(net, perturbed_image_fgsm)


# We now visualize the original and perturbed images, and see that the perturbed image is predicted incorrectly.

# In[9]:


image_show(image, pred + " " + str(score.item()))
image_show(unnormalize(perturbed_image_fgsm), new_pred_fgsm + " " + str(score_fgsm.item()))


# We now perform a similar attack using Projected Gradient Descent (PGD). PGD is an iterated version of FGSM, making multiple steps based on gradient sign, bounded by a fixed L2 or Linf norm.
# 
# In this example, we use cross-entropy loss rather than the default log-loss, and also target this attack to predict the ship class.

# In[11]:


pgd = PGD(net, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker

perturbed_image_pgd = pgd.perturb(inputs=image, radius=0.13, step_size=0.02, 
                                  step_num=7, target=torch.tensor([8]), targeted=True) 
new_pred_pgd, score_pgd = get_prediction(net, perturbed_image_pgd)

# In[12]:


image_show(image, pred + " " + str(score.item()))
image_show(unnormalize(perturbed_image_pgd.detach()), new_pred_pgd + " " + str(score_pgd.item()))


# As seen above, the perturbed input is classified as a ship, confirming the targeted attack was successful. 

# ## Robustness Metrics

# ### Attack Comparisons

# In addition to adversarial attacks, we have developed an AttackComparator, which allows quantifying model performance against any set of perturbations or attacks, including custom transformations.
# 
# In this section, we will use the AttackComparator to measure how this model performs against the FGSM / PGD attacks described above as well as torchvision transforms. Note that the AttackComparator can be used with any perturbation or attack functions.

# We will first define the desired metric function, which calculates the desired metrics we would like to evaluate and compare for different attacks. The metric function takes the model output as well as any other arguments necessary, such as the target label.

# In[13]:


import collections

ModelResult = collections.namedtuple('ModelResults', 'accuracy logit softmax')

def metric(model_out, target):
    if isinstance(target, int):
        target = torch.tensor([target])
    reshaped_target = target.reshape(len(target), 1)
    logit = torch.gather(model_out, 1, reshaped_target).detach()
    _, pred = torch.max(model_out, dim=1)
    acc = (pred == target).float()
    softmax_score =torch.gather(softmax(model_out), 1, reshaped_target).detach()
    return ModelResult(accuracy=acc, logit=logit, softmax=softmax_score)

# We can now import and initialize AttackComparator with the model, metric, and preprocessing functions. We then add the desired perturbations we want to evaluate, including FGSM, random rotations, and Gaussian blur.

# In[14]:


from captum.robust import AttackComparator

comparator = AttackComparator(forward_func=net, metric=metric, preproc_fn=normalize)

# In[15]:


comparator.add_attack(transforms.RandomRotation(degrees=30), "Random Rotation", num_attempts=100)
comparator.add_attack(transforms.GaussianBlur(kernel_size=3), "Gaussian Blur", num_attempts=1)
comparator.add_attack(fgsm, "FGSM", attack_kwargs={"epsilon": 0.15}, 
                      apply_before_preproc=False, additional_attack_arg_names=["target"], num_attempts=1)

# We can now run the comparison for the truck image we looked at previously.

# In[16]:


comparator.evaluate(image, target=label) # perturbations_per_eval can be set to > 1 for improved performance

# From these results, we see that random rotations generally lead to a correct prediction, but the worst-case rotation still led to a misclassified result as well as the FGSM attack.

# The comparator also allows us to aggregate results over a series of batches. We start by resetting the stored metrics from this example, and evaluate a series of batches from the test dataset. Once complete, we can look at the summary returned by the AttackComparator.

# In[17]:


comparator.reset()

# In[18]:


n_batches = 100
for i, (batch, batch_label) in enumerate(testloader):
    if i > n_batches:
        break
    comparator.evaluate(batch, target=batch_label, perturbations_per_eval=50)

# In[19]:


comparator.summary()

# ### Minimal Perturbation

# Next, we would like to find the minimal perturbation that causes misclassification, e.g. what is the minimum blurring of an image that causes it to be misclassified? What does this counterfactual image look like?

# In order to do this, we first implement a Gaussian blur function parametrized by the kernel size, using the corresponding torchvision transform.

# In[20]:


def gaussian_blur(image, kernel_size):
    blur = transforms.GaussianBlur(kernel_size)
    return blur(image)

# We can now import and construct the MinimalPerturbation object, providing the model, attack, and parameter to test. We provide the range of kernel sizes between 1 and 31, setting the step to 2 to include only odd numbers.

# In[21]:


from captum.robust import MinParamPerturbation

# In[22]:


# By default MinimalPerturbation compares the argmax index of the model output with target to determine
# correctness. If another definition of correctness is preferred, a custom correctness function can be provided.
min_pert = MinParamPerturbation(forward_func=net, attack=gaussian_blur, arg_name="kernel_size", mode="linear",
                               arg_min=1, arg_max=31, arg_step=2,
                               preproc_fn=normalize, apply_before_preproc=True, num_attempts=5)

# Calling evaluate with the given image returns the perturbed image as well as minimum kernel size needed for the model to misclassify the image.

# In[23]:


# None is returned if no choice of argument leads to an incorrect prediction
alt_im, kernel_size = min_pert.evaluate(image, target=label)
print("Minimum Kernel Size for Misclassification: ", kernel_size)

# We see that a kernel size of 7 was the minimum necessary to misclassify this image. Let's look at the perturbed image and corresponding prediction, and how this compares with the original.

# In[24]:


# Blurred Image
new_pred_blur, score_blur = get_prediction(net, alt_im, normalize_im=True)
image_show(alt_im, new_pred_blur + " " + str(score_blur.item()))

# Original
image_show(image, pred + " " + str(score.item()))


# We see that the blurred image is now predicted as a car rather than a truck.

# We can also utilze this perturbation metric in conjunction with attribution results to identify a counterfactual example with the minimum number of of pixels ablated (based on attribution ordering) to misclassify the image.

# Let's first obtain attribution results using FeatureAblation, using a feature mask which groups 4 x 4 squares of pixels.

# In[25]:


feature_mask = torch.arange(64).reshape(8,8).repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0).reshape(1,1,32,32)
print(feature_mask)

# In[26]:


from captum.attr import FeatureAblation

ablator = FeatureAblation(net)
attr = ablator.attribute(normalize(image), target=label, feature_mask=feature_mask)
# Choose single channel, all channels have same attribution scores
pixel_attr = attr[:,0:1]

# We now define a method which perturbs the image to dropout a given number of pixels based on feature ablation results.

# In[27]:


def pixel_dropout(image, dropout_pixels):
    keep_pixels = image[0][0].numel() - int(dropout_pixels)
    vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
    return (pixel_attr < vals.item()) * image

# We create a minimal perturbation object to find the minimum value of dropout_pixels needed for misclassification. We can also use binary mode rather than linear, which performs binary search between the given min and max ranges.

# In[28]:


min_pert_attr = MinParamPerturbation(forward_func=net, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                     arg_min=0, arg_max=1024, arg_step=16,
                                     preproc_fn=normalize, apply_before_preproc=True)

# In[29]:


pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(image, target=label, perturbations_per_eval=10)
print("Minimum Pixels Dropped:", pixels_dropped)

# Let's now take a look at this counterfactual example and corresponding prediction.

# In[30]:


# Feature Dropout Image
new_pred_dropout, score_dropout = get_prediction(net, pixel_dropout_im, normalize_im=True)
image_show(pixel_dropout_im, new_pred_dropout + " " + str(score_dropout.item()))


# Removed portions include the front of the truck as well as some portions of the background, causing the model to misclassify the truck as a frog.
