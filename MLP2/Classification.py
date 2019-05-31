import sys
from os import getcwd
from os import listdir
from os.path import isfile, join
import numpy as np
import nibabel as nib
from sklearn.svm.classes import SVC
from sklearn.svm.classes import NuSVC
import re
from scipy import linalg as LA
import scipy as sp
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    '''
    Helper function to sort files by their name numerically, so that file_7 comes before file_10
    '''
    #Spliting by numeric characters prsent
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def cubeMeanVoxels(path):
    x_s = 50                                              #Best value 40 
    x_e = 110                                             #Best value 120
    y_s = 50                                              #Best value 80
    y_e = 110                                             #Best value 160
    z_s = 50                                              #Best value 40
    z_e = 110                                             #Best value 120
    cube_edge_x = 4                                      #Best value 10
    cube_edge_y = 4
    cube_edge_z = 4
    num_cubes = ((x_e - x_s) * (y_e - y_s) * (z_e - z_s)) // (cube_edge_x * cube_edge_y * cube_edge_z)
    feaMat = np.empty((0, num_cubes), float)

    # first path leads to a .DS_store file
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        
        # For each dimension, picking the range where maximum voxel intensity is present.
        # For most of the training images, it is within cube selected below
        arr = arr[x_s:x_e, y_s:y_e, z_s:z_e]
        arr = arr.reshape((x_e - x_s, y_e - y_s, z_e - z_s))
        
        fea = np.empty((num_cubes,), float)
        index = 0
        for i in range(0, x_e - x_s, cube_edge_x):
            for j in range(0, y_e - y_s, cube_edge_y):
                for k in range(0, z_e - z_s, cube_edge_z):
                    #print(np.mean(arr[i:i+cube_edge, j:j+cube_edge, k:k+cube_edge]))

                    fea[index] = np.mean(arr[i:i+cube_edge_x, j:j+cube_edge_y, k:k+cube_edge_z])
                    #fea[index] = np.var(arr[i:i+cube_edge_x-1, j:j+cube_edge_y-1, k:k+cube_edge_z-1])
                    #fea[index] = sp.stats.chisquare(
                        #arr[i:i+cube_edge, j:j+cube_edge, k:k+cube_edge].reshape( ( cube_edge * cube_edge * cube_edge) ) )[1]

                    index = index + 1
        feaMat = np.append(feaMat, fea.reshape((1, fea.shape[0])), axis=0)
    return feaMat

def feature_selection(X, y):
    k_best = SelectKBest(chi2, k=10).fit(X[:278], y)
    X_new = k_best.transform(X)

    #X_new = VarianceThreshold(threshold=12000).fit_transform(X)
    return X_new

def mutualInformationVoxels(path):
    x_s = 40                                              #Best value 40 
    x_e = 120                                             #Best value 120
    y_s = 40                                              #Best value 80
    y_e = 120                                             #Best value 160
    z_s = 40                                              #Best value 40
    z_e = 120                                             #Best value 120
    cube_edge_x = 16                                      #Best value 10
    cube_edge_y = 16
    cube_edge_z = 16
    num_cubes = ( ((x_e - x_s) * (y_e - y_s)) // (cube_edge_x*cube_edge_y) ) * ( ((z_e - z_s) // cube_edge_z) - 1 )
    feaMat = np.empty((0, num_cubes), float)

    # first path leads to a .DS_store file
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        
        # For each dimension, picking the range where maximum voxel intensity is present.
        # For most of the training images, it is within cube selected below
        arr = arr[x_s:x_e, y_s:y_e, z_s:z_e]
        arr = arr.reshape((x_e - x_s, y_e - y_s, z_e - z_s))
        
        fea = np.empty((num_cubes,), float)
        index = 0
        for i in range(0, x_e - x_s, cube_edge_x):
            for j in range(0, y_e - y_s, cube_edge_y):
                for k in range(0, z_e - z_s - cube_edge_z, cube_edge_z):
                    fea[index] = normalized_mutual_info_score(
                        arr[i:i+cube_edge_x, j:j+cube_edge_y, k:k+cube_edge_z]
                        .reshape( ( cube_edge_x * cube_edge_y * cube_edge_z) 
                                 ),
                        arr[i:i+cube_edge_x, j:j+cube_edge_y, k+cube_edge_z:k+(2*cube_edge_z)]
                        .reshape( ( cube_edge_x * cube_edge_y * cube_edge_z) )
                        )

                    index = index + 1
        feaMat = np.append(feaMat, fea.reshape((1, fea.shape[0])), axis=0)
    return feaMat

def readVoxelsFlat(path):
    '''
    Read voxel values from nii images.
    Returns an array of features for all images present in the path passed as argument.
    '''
    l = np.empty((0, 216000), int)
    # first path leads to a .DS_store file
    for f in path:
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
    l = np.empty((0,80**3))
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        l = np.append(l, arr[40:120, 80:160, 40:120].reshape(80**3))
        #For each dimension, picking the range where maximum voxel intensity is present. 
        #For most of the training images, it is within cude selected below
        #arr = arr[40:120, 80:160, 40:120]
        #flat = arr.reshape((80 * 80 * 80))
        #fea = np.zeros((4419))
        
        ##Counting the number of voxels for each intensity value present in the cube selected above
        #for val in flat:
        #    if val <= 4418:
        #        fea[val] = fea[val] + 1

        #l = np.append(l, fea.reshape((1, 4419)), axis=0)
    return l

def maxVoxel(path):
    maxVox = -10
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        m = np.max(arr)
        if maxVox < m:
            maxVox = m
        #For each dimension, picking the range where maximum voxel intensity is present. 
        #For most of the training images, it is within cude selected below
        #arr = arr[70:100,90:120,70:100]
        #flat = arr.reshape((176*208*176))
        
        #maxVox = -10
        #Counting the number of voxels for each intensity value present in the cube selected above
        #for val in flat:
        #    if val > maxVox:
        #        maxVox = val
    return maxVox

def logloss(act, pred):
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    ll = act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred))
    ll = ll * -1.0
    return ll

def cv_split(totalSamples, totalLabels, K):
    '''
    takes total samples with labels
    returns (samples to train with labels, samples to validate with labels)
    80 - 20 break
    '''
    num = len(totalLabels)
    shuffle_index = np.random.permutation(num)
    shuffled_samples = totalSamples[shuffle_index]
    shuffled_labels = totalLabels[shuffle_index]

    fold_samples = []
    fold_labels = []

    for i in range(0, num, int(np.ceil(num/K)) ):
        fold_samples.append(shuffled_samples[i : i + int(np.ceil(num/K)) ])
        fold_labels.append(shuffled_labels[i : i + int(np.ceil(num/K)) ])

    return (fold_samples, fold_labels)

if __name__ == '__main__':
    '''
    Parameters to pass: 
    First - Path to folder containing set_train, set_test and targets.csv
    Second - Path to folder where final csv file to be stored with name Dota2Prediction.csv
    '''
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]

    mode = 'validate'

    #train set
    trainDir = inputDir + "/set_train"
    totalTrainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)
 
    #Build feature vectors for training images
    samples = cubeMeanVoxels(totalTrainFiles)
    
    #samples = readVoxels(totalTrainFiles)

    print(len(samples))

    targetsPath = inputDir + "/targets.csv" 
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    #Read age labels of training images
    labels = []
    for t in targets:
        labels.append(t[0])

    testDir = inputDir + "/set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)
    #Read feature vectors of test images
    testSamples = cubeMeanVoxels(testFiles)
    #testSamples = feature_selection(testSamples, np.array(labels))

    samples = feature_selection(np.concatenate([samples, testSamples], axis = 0), np.array(labels))
    print("Number of features = {}".format(samples[0].shape))

    samples, testSamples = samples[:278], samples[278:]

    if mode == 'validate':
        # K(=10) FOLD CROSS VALIDATION
        K = 10
        fold_samples, fold_labels = cv_split(samples, np.array(labels), K)
        log_loss = [['Log Loss'],[]]
        total_ll = 0.0
        for fold in range(K):
            samples_chunk = fold_samples[:fold] + fold_samples[fold+1:]
            labels_chunk = fold_labels[:fold] + fold_labels[fold+1:]

            #Training L1 logistic regression
            logRegrL1 = linear_model.LogisticRegression(C=1, penalty='l1')
            logRegrL1.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

            #logRegrL2 = linear_model.LogisticRegression(C=0.1, penalty='l2')
            #logRegrL2.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

            #Training SVM with linear kernel
            #svmRbf = SVC(kernel='rbf', probability=True)
            #svmRbf.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

            #Training SVM with linear kernel
            svmLin = SVC(kernel='linear', probability=True)
            svmLin.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

            #Training Random Forest Classifier
            rfc = RandomForestClassifier(n_estimators=100)
            rfc.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

            #TEST ON CROSS VALIDATION HOLD OUT SET
            val = [i for i in range(len(fold_labels[fold]))]
            id = 0
            for item in fold_samples[fold]:
                predictionL1 = logRegrL1.predict_proba(item)#first component is probability of 0 class, second is of class 1

                #predictionL2 = logRegrL2.predict_proba(item)

                predictionSvmLin = svmLin.predict_proba(item)

                #predictionSvmRbf = svmRbf.predict_proba(item) 

                predictionRfc = rfc.predict_proba(item)

                #Taking the average of each of the model predictions as final age prediction
                val[id] = (predictionL1[0][1] + predictionSvmLin[0][1] + predictionRfc[0][1])/3.0
                id = id + 1

            
            for i in range(len(fold_labels[fold])):
                total_ll += logloss(fold_labels[fold][i], val[i])
    

        log_loss[1] = total_ll/len(samples)
        #Save csv file in the output directory provided as argument with name Dota2Prediction.csv
        np.savetxt(outputDir + "/Dota2Val.csv", 
               log_loss,
               delimiter=',', 
               fmt='%s'
               )

        print("Validation complete!")

    else:
        #Training L1 logistic regression
        logRegrL1 = linear_model.LogisticRegression(C=1, penalty='l1')
        logRegrL1.fit(samples, labels)

        #logRegrL2 = linear_model.LogisticRegression(C=0.1, penalty='l2')
        #logRegrL2.fit(samples, labels)

        #Training SVM with linear kernel
        #svmRbf = SVC(kernel='rbf', probability=True)
        #svmRbf.fit(samples, labels)

        #Training SVM with linear kernel
        svmLin = SVC(kernel='linear', probability=True)
        svmLin.fit(samples, labels)

        #Training Random Forest Classifier
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(samples, labels)

        #gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        #gbc.fit(samples, labels)

        #lda = LinearDiscriminantAnalysis()
        #lda.fit(samples, labels)

        #mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        #mlp.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )

        #p = Perceptron()
        #p.fit(samples, labels)

        #sgd = SGDClassifier(loss="log", penalty="l2")
        #sgd.fit(samples, labels)

        #test set
        #testDir = inputDir + "\\set_test"
        #testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

        #Read feature vectors of test images
        #testSamples = cubeMeanVoxels(testFiles)
        #testSamples = feature_selection(testSamples, np.array(labels))
        #testSamplesFlat = readVoxelsFlat(testFiles)
        #testSamplesVar = testSamplesFlat[:, ind]
        #print(len(testSamples))

        #2D array to report final prediction in format (ID,Prediction)
        final = [[0 for j in range(2)] for i in range(139)]
        final[0][0] = 'ID'
        final[0][1] = 'Prediction'
        id = 1

        #Predict age of test image using each of the 4 models trained above
        for item in testSamples:
            predictionL1 = logRegrL1.predict_proba(item)#first component is probability of 0 class, second is of class 1
            #predictionL2 = logRegrL2.predict_proba(item)
            predictionSvmLin = svmLin.predict_proba(item)
            #predictionSvmRbf = svmRbf.predict_proba(item) 
            predictionRfc = rfc.predict_proba(item)
            #predictionGbc = gbc.predict_proba(item)
            #predictionLda = lda.predict_proba(item)
            #predictionP = p.predict(item)
            #predictionSgd = sgd.predict_proba(item)
            #predictionMlp = mlp.predict_proba(item)


            final[id][0] = id
            #Taking the average of each of the model predictions as final age prediction
            final[id][1] = (predictionL1[0][1] + predictionSvmLin[0][1] + predictionRfc[0][1])/3.0
            id = id + 1
    
        #Save csv file in the output directory provided as argument with name Dota2Prediction.csv
        np.savetxt(outputDir + "/Dota2Prediction.csv", 
               final,
               delimiter=',', 
               fmt='%s'
               )
        print("Finished!")