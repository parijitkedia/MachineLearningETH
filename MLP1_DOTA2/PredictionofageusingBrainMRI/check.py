from os import getcwd
from os import listdir
from os.path import isfile, join
import numpy as np
import nibabel as nib
from sklearn.svm.classes import SVC
import re
from scipy import linalg as LA
from sklearn import linear_model
from sklearn.decomposition import PCA

#def PCA(data, dims_rescaled_data=2):
#    """
#    returns: data transformed in 2 dims/columns + regenerated original data
#    pass in: data as 2D NumPy array
#    """
#    import numpy as NP
#    from scipy import linalg as LA
#    m, n = data.shape
#    # mean center the data
#    data -= data.mean(axis=0)
#    # calculate the covariance matrix
#    R = NP.cov(data, rowvar=False)
#    # calculate eigenvectors & eigenvalues of the covariance matrix
#    # use 'eigh' rather than 'eig' since R is symmetric, 
#    # the performance gain is substantial
#    evals, evecs = LA.eigh(R)
#    # sort eigenvalue in decreasing order
#    idx = NP.argsort(evals)[::-1]
#    evecs = evecs[:,idx]
#    # sort eigenvectors according to same index
#    evals = evals[idx]
#    # select the first n eigenvectors (n is desired dimension
#    # of rescaled data array, or dims_rescaled_data)
#    evecs = evecs[:, :dims_rescaled_data]
#    # carry out the transformation on the data using eigenvectors
#    # and return the re-scaled data, eigenvalues, and eigenvectors
#    return NP.dot(evecs.T, data.T).T, evals, evecs

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if __name__ == '__main__':

    #train set
    curDir = getcwd()
    trainDir = curDir + "/data/set_train"
    trainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)

    l = np.empty((0,176*208*176))
  
    index = 0
    for f in trainFiles:
        img = nib.load(f)
        arr = img.get_data()
        #arr = arr[40:110,80:150,40:110]
        #mean = np.zeros(shape=(160,140))
        ##for i in range(len(arr)):
        ##    #mean = np.add(mean, arr[i].reshape(160,140))
        ##    #l = np.vstack([l,arr[i].reshape(1, 160*140)])
        ##    #l = np.append(l, arr[i].reshape(1, 160*140), axis=0)
        ##    l[index] = arr[i].reshape(1, 50*50)
        ##    index = index + 1

        l = np.append(l, arr.reshape(1,176*208*176), axis=0)

        #mean = (1/130) * mean
        #l.append(mean)

    #samples = []
    #for item in l:
    #    mean = np.zeros(shape=(140))
    #    for row in range(160):
    #        mean = np.add(mean, item[row])
    #    mean = (1/160) * mean
    #    samples.append(mean)

    print(len(l))

    samples = l

    #PCA on training samples
    #(a,b,c) = PCA(samples, 15)
    #samples = a

    pca = PCA(n_components=50)
    pca.fit(samples)

    samples = pca.transform(samples)

    targetsPath = '/targets.csv'
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    labels = []
    for t in targets:
        #for i in range(50):
        labels.append(t[0])

    #SVM 
    #svc = SVC(kernel='linear')
    #svc.fit(samples, labels)

    #linear regression
    regr = linear_model.LinearRegression()
    regr.fit(samples, labels)


    #test set
    testDir = curDir + "/data/set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

    l = np.empty((0,176*208*176))
    for f in testFiles:
        img = nib.load(f)
        arr = img.get_data()
        #arr = arr[40:110,80:150,40:110]
        #mean = np.zeros(shape=(160,140))
        #for i in range(len(arr)):
        #    #mean = np.add(mean, arr[i].reshape(160,140))
        #    l = np.append(l, arr[i].reshape(1, 50*50), axis=0)

        l = np.append(l, arr.reshape(1,176*208*176), axis=0)

        #mean = (1/130) * mean
        #l.append(mean)

    #testSamples = []
    #for item in l:
    #    mean = np.zeros(shape=(140))
    #    for row in range(160):
    #        mean = np.add(mean, item[row])
    #    mean = (1/160) * mean
    #    testSamples.append(mean)

    testSamples = l

    print(len(testSamples))

    #testSamples = np.vstack(testSamples)

    #PCA on test samples
    #(a,b,c) = PCA(testSamples, 15)
    #testSamples = a

    pca = PCA(n_components=50)
    pca.fit(testSamples)
    testSamples = pca.transform(testSamples)


    final = []
    for item in testSamples:
        #SVM prediction
        #prediction = svc.predict(item)

        #linear regression prediction
        prediction = regr.predict(item)
        final.append(prediction)

    #age = []
    #index = 0
    #while(index < len(final)):
    #    count = 0
    #    mean = 0
    #    while(count < 50):
    #        mean = mean + final[index]
    #        count = count + 1
    #        index = index + 1
    #    mean = (1/50) * mean
    #    age.append(mean)

    np.savetxt('mydata.csv', 
           final, 
           delimiter=',', 
           fmt='%3i', 
           header='Results')