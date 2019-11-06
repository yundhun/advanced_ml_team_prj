
#SVM
#https://github.com/whimian/SVM-Image-Classification.git

#DL
#
import numpy as np
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.transform import resize
import skimage
import pandas as pd



def load_image_files(file_list, labels, dimension=(64, 64)):
    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, file in enumerate(file_list):
        img = skimage.io.imread(file)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(labels[i])
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 images=images,
                 DESCR=descr)

tr_csv = pd.read_csv('images/boneage-training-dataset.csv')
tr_file_names = [ 'images/rsna-bone-age/boneage-training-dataset/' + str(s) + '.png' for s in list(tr_csv.loc[:100,'id']) ]
tr_labels = list( tr_csv.loc[:100,'boneage'] )

image_dataset = load_image_files(tr_file_names, tr_labels)

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))