from os import getcwd
from os import listdir
from os.path import isfile, join
import numpy as np
import nibabel as nib
from sklearn.svm.classes import SVC
import re
from scipy import linalg as LA

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == '__main__':

    #train set
    curDir = getcwd()
    trainDir = curDir + "\\Docs\\set_train"
    trainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)

    l = []    
    for f in trainFiles:
        img = nib.load(f)
        arr = img.get_data()
        mean = np.zeros(shape=(208,176))
        for i in range(len(arr)):
            mean = np.add(mean, arr[i].reshape(208,176))
        mean = (1/176) * mean
        l.append(mean)

    samples = []
    for item in l:
        mean = np.zeros(shape=(176))
        for row in range(208):
            mean = np.add(mean, item[row])
        mean = (1/208) * mean
        samples.append(mean)

    print(len(samples))

    samples = np.vstack(samples)

    #PCA on training samples
    (a,b,c) = PCA(samples, 100)
    samples = a

    targetsPath = 'C:\\Users\\ankit\\Desktop\\Eth\\Autumn Semester 2016\\Machine Learning\\Project\\Project 1\\Slicing\\Slicing\\Docs\\targets.csv'
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    labels = []
    for t in targets:
        labels.append(t[0])

    svc = SVC(kernel='linear')
    svc.fit(samples, labels)

    #test set
    testDir = curDir + "\\Docs\\set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

    l = []    
    for f in testFiles:
        img = nib.load(f)
        arr = img.get_data()
        mean = np.zeros(shape=(208,176))
        for i in range(len(arr)):
            mean = np.add(mean, arr[i].reshape(208,176))
        mean = (1/176) * mean
        l.append(mean)

    testSamples = []
    for item in l:
        mean = np.zeros(shape=(176))
        for row in range(208):
            mean = np.add(mean, item[row])
        mean = (1/208) * mean
        testSamples.append(mean)

    print(len(testSamples))

    testSamples = np.vstack(testSamples)

    #PCA on test samples
    (a,b,c) = PCA(testSamples, 100)
    testSamples = a


    final = []
    for item in testSamples:
        prediction = svc.predict(item)
        final.append(prediction)

    np.savetxt('mydata.csv', 
           final, 
           delimiter=',', 
           fmt='%3i', 
           header='Results')

    