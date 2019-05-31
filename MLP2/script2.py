import sys
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    '''
    Helper function to sort files by their name numerically, so that file_7 comes before file_10
    '''
    # Spliting by numeric characters present
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def readVoxelsFlat(path):
    '''
    Read voxel values from nii images.
    Returns an array of features for all images present in the path passed as argument.
    '''
    l = np.empty((0, 216000), int)
    # first path leads to a .DS_store file
    for f in path[1:]:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        # For each dimension, picking the range where maximum voxel intensity is present.
        # For most of the training images, it is within cube selected below
        arr = arr[40:100, 100:160, 50:110]
        flat = arr.reshape((60 * 60 * 60))
        l = np.append(l, flat.reshape((1, 216000)), axis=0)
    return l

def readVoxels(path):
    '''
    Read voxel values from nii images. The max voxel value present in any train image is 4419.
    Hence, counting the number of different voxels with intensities in between 0 and 4419 and using 
    that as features.
    Returns an array of features for all images present in the path passed as argument.
    '''
    l = np.empty((0, 4419))
    # first path leads to a .DS_store file
    for f in path[1:]:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        # For each dimension, picking the range where maximum voxel intensity is present.
        # For most of the training images, it is within cube selected below
        arr = arr[40:100, 100:160, 50:110]
        flat = arr.reshape((60 * 60 * 60))
        fea = np.zeros((4419))

        # Counting the number of voxels for each intensity value present in the cube selected above
        for val in flat:
            if val <= 4418:
                fea[val] = fea[val] + 1

        l = np.append(l, fea.reshape((1, 4419)), axis=0)
    return l



    inputDir = "/Users/Simon/Documents/Master/ETH/Machine Learning/Project/data"
    outputDir = "/Users/Simon/Documents/Master/ETH/Machine Learning/Project/project2"

    # train set
    trainDir = inputDir + "/set_train"
    trainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)

    # Build feature vectors with aggregated number of voxels
    samples = readVoxels(trainFiles)

    # Build feature vectors by selecting the voxels with the most variance over all training samples
    samplesFlat = readVoxelsFlat(trainFiles)
    varArr = np.apply_along_axis(np.var, 0, samplesFlat)
    ind = varArr>200000
    sum(ind) # check how many features to avoid overfit; for threshold var>300000  best scores
    samplesVar = samplesFlat[:,ind]

    print(len(samples))

    targetsPath = inputDir + "/targets2.csv"
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    # Read age labels of training images
    labels = []
    for t in targets:
        labels.append(t[0])

    # test set
    testDir = inputDir + "/set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

    # Read feature vectors of test images with aggregated number of voxels
    testSamples = readVoxels(testFiles)
    print(len(testSamples))

    # Read feature vectors of test images selecting the voxels with the most variance over all training samples
    testSamplesFlat = readVoxelsFlat(testFiles)
    testSamplesVar = testSamplesFlat[:, ind]





#####################################################################################################################

    # Training LASSO regressor, alpha value tuned to produce best result when used alone on the test set
    regrLog = linear_model.LogisticRegression(C=0.1, penalty='l2')
    regrLog.fit(samples, labels)

    regrLogL1 = linear_model.LogisticRegression(C=1, penalty='l1')
    regrLogL1.fit(samples, labels)

    regrLogVar = linear_model.LogisticRegression(C=0.001, penalty='l1')
    regrLogVar.fit(samplesVar, labels)

    svc = svm.SVC(kernel='linear', probability=True)
    svc.fit(samples, labels)

    svcVar = svm.SVC(kernel='linear', probability=True)
    svcVar.fit(samplesVar, labels)

    #####################################################################################################################
    # LogL1

    # 2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(2)] for i in range(139)]
    final[0][0] = 'ID'
    final[0][1] = 'Prediction'
    id = 1

    # Predict age of test image using each of the 4 models trained above
    for item in testSamples:
        predictionL1 = regrLogL1.predict_proba(item)
        final[id][0] = id
        # Taking the average of each of the model predictions as final age prediction
        final[id][1] = predictionL1[0][1]
        id = id + 1

    # Save csv file in the output directory provided as argument with name Dota2Prediction.csv
    np.savetxt(outputDir + "/dota2submission.csv",
               final,
               delimiter=',',
               fmt='%s'
               )
    print("Finished!")


#####################################################################################################################
    # LogL1 var


    # 2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(2)] for i in range(139)]
    final[0][0] = 'ID'
    final[0][1] = 'Prediction'
    id = 1
    for item in testSamplesVar:
        predictionLogVar = regrLogVar.predict_proba(item)
        final[id][0] = id
        # Taking the average of each of the model predictions as final age prediction
        final[id][1] = predictionLogVar[0][1]
        id = id + 1

    # Save csv file in the output directory provided as argument with name Dota2Prediction.csv
    np.savetxt(outputDir + "/dota2submissionVar.csv",
               final,
               delimiter=',',
               fmt='%s'
               )
    print("Finished!")


#####################################################################################################################
    # SVC

    # 2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(2)] for i in range(139)]
    final[0][0] = 'ID'
    final[0][1] = 'Prediction'
    id = 1

    for item in testSamples:
        predictionSVC = svc.predict_proba(item)
        final[id][0] = id
        # Taking the average of each of the model predictions as final age prediction
        final[id][1] = predictionSVC[0][1]
        id = id + 1

    # Save csv file in the output directory provided as argument with name Dota2Prediction.csv
    np.savetxt(outputDir + "/dota2submissionSVC.csv",
               final,
               delimiter=',',
               fmt='%s'
               )
    print("Finished!")

#####################################################################################################################

    # 2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(2)] for i in range(139)]
    final[0][0] = 'ID'
    final[0][1] = 'Prediction'
    id = 1

    for item in testSamplesVar:
        predictionSVCVar = svcVar.predict_proba(item)
        final[id][0] = id
        # Taking the average of each of the model predictions as final age prediction
        final[id][1] = predictionSVCVar[0][1]
        id = id + 1

    # Save csv file in the output directory provided as argument with name Dota2Prediction.csv
    np.savetxt(outputDir + "/dota2submissionSVCVar.csv",
               final,
               delimiter=',',
               fmt='%s'
               )
    print("Finished!")


#####################################################################################################################

