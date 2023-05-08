# EMNIST-Classification-Problem---MLP---CNN
Project's aim is to build a neural network model to solve the EMNIST classification problem.

Import necessary libraries
# Import necessary libraries
!pip install tensorflow-gpu
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import random
import tensorflow_datasets as tfds 
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
!pip install keras-tuner
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from kerastuner.tuners import RandomSearch
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

