#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import random
import csv
import cv2
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

train_images = []
with open('input/train/train2.csv', 'rt') as file:
    data = csv.reader(file)
    for row in data:
        img = row[0]
        train_images.append(img)
random.shuffle(train_images)

test_images = []
with open('input/test_ApKoW4T.csv', 'rt') as file:
    data = csv.reader(file)
    for row in data:
        img = row[0]
        test_images.append(img)
random.shuffle(test_images)


# In[2]:


nrows = 150
ncols = 150
channels = 3

def read_process_image(image_list):
    X = [] # for resized images
    y = [] # for labels
    
    for image in image_list:
        exc = ""
        image_path = "input/train/images-5-class/" + image
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            X.append(cv2.resize(img, (nrows, ncols), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            exc = image
            print(str(e))
            print(image_path)
        with open('input/train/train2.csv', 'rt') as file:
            data = csv.reader(file)
            for row in data:
                if row[0] == image:
                    label = row[1]
        if (exc != image):
            y.append(label)
    
    return X, y


# In[3]:


X, y = read_process_image(train_images)


# In[4]:


train_files = pd.read_csv('input/train/train2.csv', names=['image', 'category'])
train_files.head()


# In[5]:


test_files = pd.read_csv('input/test_ApKoW4T.csv', names=['image'])
test_files.head()


# In[6]:


plt.figure(figsize=(20,10))
columns = 5
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[7]:


X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels ships')


# Labels
# - 1: Cargo
# - 2: Military
# - 3: Carrier
# - 4: Cruise
# - 5: Tankers

# In[8]:


print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[9]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)


# In[11]:


import gc

gc.collect()

ntrain = len(X_train)
nval = len(X_val)

batch_size = 32


# In[12]:


from keras.applications import VGG19

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base = ResNet152V2(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base = NasNetLarge(weights='imagenet', include_top=False, input_shape=(150,150,3))
# conv_base = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(150,150,3))


# In[13]:


conv_base.summary()


# In[40]:


from keras import layers
from keras import models
from keras.layers import Dropout

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(Dropout(0.2, input_shape=(150,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))


# In[41]:


model.summary()


# In[42]:


print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))


# In[43]:


from keras import optimizers

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])


# In[44]:


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)


# In[45]:


train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# In[46]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=50,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)


# In[47]:


model.save_weights('models/model_weights_vgg19_5.h5')
# model.save_weights('models/model_weights_inception_resnet_v2.h5')
model.save('models/model_keras_vgg19_5.h5', include_optimizer=False) # to get rid of load_model 'saved optimizer' warning
# model.save('models/model_keras_inception_resnet_v2.h5', include_optimizer=False)


# In[62]:


import pydot
from keras.utils import plot_model
from IPython.display import Image
plot_model(model, to_file='model.png', show_shapes= True)
Image("model.png")


# In[63]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()


# In[64]:


def smooth_plot(points, factor=0.7):
    smooth_pts = []
    for point in points:
        if smooth_pts:
            previous = smooth_pts[-1]
            smooth_pts.append(previous * factor + point * (1 - factor))
        else:
            smooth_pts.append(point)
    return smooth_pts


# In[65]:


plt.plot(epochs, smooth_plot(acc), 'b', label='Training accurarcy')
plt.plot(epochs, smooth_plot(val_acc), 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.show()


# In[66]:


test_images = []
with open('input/test_ApKoW4T.csv', 'rt') as file:
    data = csv.reader(file)
    for row in data:
        img = row[0]
        test_images.append(img)


# In[67]:


nrows = 150
ncolumns = 150
channels = 3

def read_process_image(image_list):
    X = [] # for resized images
    y = [] # for labels
    
    for image in image_list:
        exc = ""
        image_path = "input/train/images-5-class/" + image
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            X.append(cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        except Exception as e:
            exc = image
            print(str(e))
    
    return X, y


# In[68]:


X_test, y_test = read_process_image(test_images[0:10])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[69]:


from keras.models import load_model
model = load_model('models/model_keras_vgg19_5.h5')


# In[70]:


i = 0
columns = 5
text_labels = []
convert_label_dict = {'0': 'Cargo', 
                      '1': 'Military', 
                      '2': 'Carrier', 
                      '3': 'Cruise', 
                      '4': 'Tankers'}
plt.figure(figsize=(30,12))

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    pred_label = np.argmax(pred)

    text_labels.append(str(pred_label))
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + convert_label_dict.get(text_labels[i]))
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break

plt.show()


# In[ ]:



