import numpy as np
import os

def opencsv(inFile, delim=','):
    if os.path.exists(inFile):
        assert isinstance(delimiter, str), 'Ensure your delimiter is a "str" type variable'
    if os.path.exists(inFile):
        try:
            csvtable = np.genfromtxt(inFile, delimiter=delim, dtype=float, names=True)
        except:
            csvtable = np.genfromtxt(inFile, dtype=float, names=True)
    else:
        raise Exception('Check whether file exists, unable to locate: {}'.format(inFile))
    return csvtable

def csv2array(csvtable)
    array = np.zeros((len(csvtable), len(csvtable[0])), dtype=float)
    for i in range(0, len(csvtable)):
        tmp = csvtable[i]
        for j in range(0, len(csvtable[i])):
            array[i, j] = csvtable[i][j]
    return array


