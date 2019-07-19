import createFeatures as cf
from keras.models import Sequential
from keras.layers import Dense, Activation


csvPath = '/Users/sid/Documents/Projects/Enigma-ML/Dataset/T1/featsTable.csv'
tbl = cf.opencsv(csvPath)
data = cf.csv2array(tbl)
