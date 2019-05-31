import os
# import numpy as np
# from nibabel.testing import data_path
import nibabel as nib
# import csv
# from skimage import data
# from sklearn import svm
# from nilearn import plotting
import numpy as np
from nipy.testing import anatfile
from nipy import load_image
from skimage import io
#filename = os.path.join(data_path,"~/Desktop/Python/Parijitkedia/ML/Prediction of age using Brain MRI/data/set_train/train1.nii")

# target=[]
# with open("targets.csv","rb") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		target.append(row)

# clf = svm.LinearSVC()
# clf.fit(img,target)

#clf.predict()

#plotting.plot_glass_brain("/Users/parijitkedia/Desktop/Python/Parijitkedia/ML/Prediction of age using Brain MRI/data/set_train/train_1.nii")


# loc = "/Users/parijitkedia/Desktop/Python/Parijitkedia/ML/Prediction of age using Brain MRI/data/set_train/train_1.nii"
# img = load_image(loc)


# loc1 = "/Users/parijitkedia/Desktop/Python/Parijitkedia/ML/Prediction of age using Brain MRI/data/set_train/train_2.nii"
# img1 = load_image(loc1)
# print img1.get_data().shape



img = io.imread("/Users/parijitkedia/Desktop/Python/Parijitkedia/ML/Prediction of age using Brain MRI/data/set_train/train_2.nii")
print img
