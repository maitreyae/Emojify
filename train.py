import numpy as np
import cv2
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D      # key component in image processing task
from keras.optimizers import Adam    # used for traning deeplearnig models
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator    # provides convinet way to generate preprocessed images to train models

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(   
    train_dir,
    target_size=(48, 48),
    batch_size=64,   # no of sample in each batch
    color_mode="grayscale",
    class_mode='categorical')   # train_generator to train module
validation_generator = val_datagen.flow_from_directory(    
    val_dir, 
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')   # Validation_generator to train module



emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape =(48, 48, 1)))  
# conv2D is keras convolution i.e filter
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))     # reduce dependency on specific neuron
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))  # used to train subsets 
emotion_model.add(Dropout(0.25))   # filters only face
emotion_model.add(Flatten())       # converts 2D to 1D 
emotion_model.add(Dense(1024, activation='relu'))  
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))  # softmax used for o/p layer of neural n/w

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,   # traning_sample/batch_size
    epochs=50,   # for no of iterations   
    validation_data=validation_generator,  # to train the module
    validation_steps=7178 // 64
)

emotion_model.save_weights('model.h5')
