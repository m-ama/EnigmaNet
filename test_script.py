import createFeatures as cf
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def classfill(dFrame, classSel, idxRange):
    """Fills missing values with means of a class
    
    Inputs
    ------
    dFrame:   Pandas dataframe to process (type: dataframe)
            
    classSel: String indicating dataframe column name containing class information
    
    idxRange: 2x1 vector indicating lower and upper bound of data to fill in dataframe
              idxRange[0] is lower bound
              idxRange[1] is upper bound

    Returns
    -------
    data:     Dataframe will all missing values filled
    """
    uniqClass = dFrame[classSel].unique()           # All unique classes
    print('...found ' + str(uniqClass.size) + ' classes')
    print('...filling missing data with class means')
    data = dFrame.loc[:, idxRange[0]:idxRange[1]]                 # Extract all numerical value from 'dBegin' onwards
    for c in uniqClass:
        classIdx = dFrame.loc[:, classSel] == c     # Index where class is uniqClass = c
        for n in range(len(data.columns)):
            nanIdx = data.iloc[:,n].isnull()           # Index missing values
            # Compute mean of class values without nans
            # Because a Series of booleans cannot be used to index a dataframe, use the values attribute
            # to extract a bool array
            mu = np.nanmean(data.iloc[classIdx.values, n])
            data.iloc[nanIdx.values, n] = mu
    dFrame.loc[:,idxRange[0]:idxRange[1]] = data
    return dFrame

# Init Variables
classSel = 'Dx'    # Class labels
dBegin = 'ICV'      # Column where actual begins
dEnd = 'R_insula_surfavg'

# Load Files
csvPath = '/Users/sid/Documents/Projects/Enigma-ML/Dataset/T1/all.csv'
dFrame = pd.read_csv(csvPath)           # Dataframe
dFrame = classfill(dFrame, classSel, [dBegin, dEnd])

data = dFrame.loc[:,dBegin:dEnd]
# Scale data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split into training and validation sets and scale
X_train, X_test, y_train, y_test = train_test_split(data, dFrame.loc[:, classSel], test_size = 0.10, random_state = 0)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Choose whether data/labels or X_Train,y_train
dataIn = X_train
labelsIn = y_train

# Initialising the ANN
model = Sequential()

# Adding the Single Perceptron or Shallow network
model.add(Dense(output_dim=64, init='uniform', activation='relu', input_dim=dataIn.shape[1]))
# Adding dropout to prevent overfitting
model.add(Dropout(p=0.1))
# Adding hidden layers
model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
# Adding the output layer
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# criterion loss and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(dataIn, labelsIn, batch_size=50, epochs=150)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Test accuracy is {}%".format(((cm[0][0] + cm[1][1])/np.sum(cm))*100))

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