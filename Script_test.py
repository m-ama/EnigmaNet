import createFeatures as cf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Load Files
csvPath = "D:/SystemFiles/siddh/Box Sync/Home-Work/featsTable.csv"
tbl = cf.opencsv(csvPath)
data = cf.csv2array(tbl)
labels = data[:,0]
data = data[:,1:]

# Split into training and validation sets and scale
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Choose whether data/labels or X_Train,y_train
dataIn = X_train
labelsIn = y_train

# Initialize Model
model = Sequential([
    Dense(1, input_shape=(data.shape[1],)),
    Activation('relu'),
])

# Initialising the ANN
classifier = Sequential()

# Adding the Single Perceptron or Shallow network
classifier.add(Dense(output_dim=128, init='uniform', activation='relu', input_dim=dataIn.shape[1]))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# criterion loss and optimizer
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = classifier.fit(dataIn, labelsIn, batch_size=100, nb_epoch=150)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Test accuracy is {}%".format(((110/114)*100)))

# Form Graph Path
pwd = os.path.dirname(os.path.realpath(__file__))
savePath = os.path.join(pwd, 'model_fit.png')

# Plot training & validation accuracy values
plt.subplot(121)
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

plt.savefig(savePath, dpi=600)