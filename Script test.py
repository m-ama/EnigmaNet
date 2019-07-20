import createFeatures as cf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os

# Load Files
csvPath = "D:/SystemFiles/siddh/Box Sync/Home-Work/featsTable.csv"
tbl = cf.opencsv(csvPath)
data = cf.csv2array(tbl)
labels = data[:,0]
data = data[:,1:]


# Initialize Model
model = Sequential([
    Dense(1, input_shape=(data.shape[1],)),
    Activation('relu'),
])

# Compile Model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(data, labels, epochs=250, batch_size=10, validation_split=0.1)

pwd = os.path.dirname(os.path.realpath(__file__))
savePath = os.path.join(pwd, 'model_fit.png')

# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

plt.savefig(savePath, dpi=600)