import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback
import matplotlib.pyplot as plt
import scipy.stats
import random
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
classSel = 'Dx'             # Class labels
dBegin = 'ICV'              # Column where data begins
dEnd = 'R_insula_surfavg'   # Column where data ends
cBegin = 'Site'             # Column where covariates/demographics begin
cEnd = 'Sex'                # Column where covariates/demographics end
fillmissing = True          # Fill missing?
harmonize = True            # Run ComBat harmonization?
dataSplit = 0.10            # Percent of data to remove for validation
nEpochs = 500               # Training number of epochs
bSize = 50                  # Training batch size
plotType = 'Normal'         # Type of ComBat graphs to save ('Histogram' or 'Normal')

# Combat Variables
if harmonize:
    batchVar = 'Site'           # Batch effect variable
    discreteVar = ['Dx','Sex']  # Variables which are categorical that you want to predict
    continuousVar = ['Age']     # Variables which are continuous that you want to predict

# Load Files
csvPath = '/Users/sid/Documents/Projects/Enigma-ML/Dataset/T1/all.csv'
dFrame = pd.read_csv(csvPath)           # Dataframe
if fillmissing:
    dFrame = classfill(dFrame, classSel, [dBegin, dEnd])
else:
    print('...skip fill missing')

# Run combat
cData = neuroCombat(data=dFrame.loc[:,dBegin:dEnd],
                      covars=dFrame.loc[:,cBegin:cEnd],
                      batch_col=batchVar,
                      discrete_cols=discreteVar,
                      continuous_cols=continuousVar)
# Scale data
scaler = StandardScaler()
cData = scaler.fit_transform(cData)


data = np.array(dFrame.loc[:, dBegin:dEnd])
# Scale data
scaler = StandardScaler()
data = scaler.fit_transform(data)


# Split into training and validation sets and scale
if harmonize:
    X_train, X_test, y_train, y_test = train_test_split(cData, dFrame.loc[:, classSel], test_size=dataSplit, random_state=0)
else:
    X_train, X_test, y_train, y_test = train_test_split(data, dFrame.loc[:, classSel], test_size=dataSplit, random_state=0)


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
history = model.fit(dataIn, labelsIn,
                    batch_size=bSize,
                    epochs=nEpochs,
                    verbose=False,
                    callbacks=[TQDMNotebookCallback()])

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Test accuracy is {}%".format(((cm[0][0] + cm[1][1])/np.sum(cm))*100))
kappa = cohen_kappa_score(y_test, y_pred)
print('Cohen' + """'""" + 's Kappa = ' + str(kappa))

# Form Graph Path
pwd = os.getcwd()
savePathModel = os.path.join(pwd, 'model_fit.png')
savePathComBat = os.path.join(pwd, 'combat.png')

# Plot training & validation accuracy values
with plt.style.context('ggplot'):
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
plt.savefig(savePathModel, dpi=600)

# Plot ComBat before & after
szSubPlot = 4                                                                   # Number of features to plot
nBins = 20                                                                      # Number of bins

uniqSites = dFrame.loc[:,'Site'].unique()
with plt.style.context('ggplot'):                                               # Plotting style
    fig, axs = plt.subplots(np.sqrt(szSubPlot).astype(int), np.sqrt(szSubPlot).astype(int))
    for axsNum, axsIdx in enumerate(axs.reshape(-1)):                                              # Iterate over subplots
        plotIdx = random.randint(0,len(dFrame.loc[:,dBegin:dEnd].columns))      # Index random headers
        for s in uniqSites:
            siteIdx = dFrame.loc[:, 'Site'] == s
            nBefore, bBefore = np.histogram(data[siteIdx.values, plotIdx],      # Bin count before
                                           bins=nBins,
                                           density=True)
            nAfter, bAfter = np.histogram(cData[siteIdx.values, plotIdx],       # Bin count after
                                         bins=nBins,
                                         density=True)

            mBefore = np.zeros((nBins,))
            mAfter = np.zeros((nBins,))
            for i  in range(len(bBefore)-1):                                    # Get median of bin edges
                mBefore[i] = np.median([bBefore[i], bBefore[i + 1]])            # Median of bin edges (before)
                mAfter[i] = np.median([bAfter[i], bAfter[i + 1]])               # Median of bin edges (after)

            siteIdx = dFrame.loc[:,'Site'] == s                                 # Extract data for a site
            muBefore = np.mean(data[siteIdx.values, plotIdx])
            muAfter = np.mean(cData[siteIdx.values, plotIdx])
            stdBefore = np.std(data[siteIdx.values, plotIdx])
            stdAfter = np.std(cData[siteIdx.values, plotIdx])
            yBefore = scipy.stats.norm.pdf(mBefore, muBefore, stdBefore)
            yAfter = scipy.stats.norm.pdf(mAfter, muAfter, stdAfter)
            if plotType == 'Histogram':
                yBefore = nBefore
                yAfter = nAfter
            elif plotType == 'Normal':
                yBefore = scipy.stats.norm.pdf(mBefore, muBefore, stdBefore)
                yAfter = scipy.stats.norm.pdf(mAfter, muAfter, stdAfter)

            axsIdx.plot(mBefore, yBefore,                                       # Plot on subplot(axsIdx) before
                              color='#3a4750',
                              alpha=0.25)

            axsIdx.plot(mAfter, yAfter,                                         # Plot on subplot(axsIdx) after
                              color='#d72323',
                              alpha=0.25)

            if axsNum == 0 or axsNum == 2:
                axsIdx.set_ylabel('% OF SUBJECTS',
                                  fontsize=6)

            axsIdx.set_xlabel(dFrame.loc[:, dBegin:dEnd].columns[plotIdx].upper(),
                              fontsize=6)

    fig.legend(['Before ComBat', 'After ComBat'],                               # Legend
               loc = 'lower right',
               ncol=2,
               fancybox=True,
               bbox_to_anchor=(0.5,-0.1))
    plt.suptitle('ComBat Harmonization: Before and After')
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.tight_layout()
plt.savefig(savePathComBat, dpi=600)