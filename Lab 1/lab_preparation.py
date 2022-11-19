# -*- coding: utf-8 -*-
"""lab_preparation.ipynb

#Πρώτη εργαστηριακή άσκηση
##Αλεξανδρόπουλος Σταμάτης (03117060) , Γκότση Πολυτίμη-Άννα (03117201)
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt
# import sys
# sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks/pattern_rec')
# sys.path.insert(0,'/content/drive/MyDrive/pattern_rec')
import lib
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.ensemble import VotingClassifier,BaggingClassifier
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F

"""##Βήμα 1"""

# f = open("/content/drive/My Drive/Colab Notebooks/pattern_rec/data/test.txt", "r")     # read test data
f = open("./data/test.txt", "r")     # read test data
content=f.readlines()
test_array = np.array([line.split() for line in content] )                             # convert test data to array
# f = open("/content/drive/My Drive/Colab Notebooks/pattern_rec/data/train.txt", "r")    # read train data
f = open("./data/train.txt", "r")    # read train data
content=f.readlines()
train_array = np.array([line.split() for line in content])                             # convert train data to array

y_train=train_array[:,:1]               # get train labels
y_train=y_train.flatten()               # convert to 1d
y_train=y_train.astype(np.float64)      # convert strings to floats
X_train=train_array[:,1:]               # get features of train data
X_train=X_train.astype(np.float64)      # convert strings to floats

y_test=test_array[:,:1]                 # get test labels
y_test=y_test.flatten()                 # convert to  1d
y_test=y_test.astype(np.float64)        # convert strings to floats
X_test=test_array[:,1:]                 # get features of test data
X_test=X_test.astype(np.float64)        # convert strings to floats

print('Dimensions of train labels array:', y_train.shape)
print('Dimensions of train samples array:', X_train.shape)
print('Dimensions of test labels array:', y_test.shape)
print('Dimensions of test samples array:', X_test.shape)

"""##Βήμα 2"""

lib.show_sample(X_train,130)        # plot digit at index 131 of train data

print('Digit is:', int(y_train[130]))

"""##Βήμα 3"""

lib.plot_digits_samples(X_train,y_train)   # plot one sample (randomly picked) for each of the digits 0-9

"""##Βήμα 4"""

mean_zero=lib.digit_mean_at_pixel(X_train,y_train,0)  # calculate mean value for pixel (10,10) (default) for digit 0 , using the train data
print("Mean value is:",mean_zero)

"""##Βήμα 5"""

mean_variance=lib.digit_variance_at_pixel(X_train,y_train,0)  # calculate variance for pixel (10,10) (default) for digit 0 , using the train data
print("Variance value is:",mean_variance)

"""##Βήμα 6"""

zero_mean_values=lib.digit_mean(X_train,y_train,0)            # calculate mean value for all features of digit 0
zero_variance_values=lib.digit_variance(X_train,y_train,0)    # calculate variance for all features of digit 0

print('Mean values for digit 0 are: ',zero_mean_values)

print('Variance values for digit 0 are: ',zero_variance_values)

"""##Βήμα 7"""

plt.imshow(np.reshape(zero_mean_values,(16,16)),cmap='gray')  # plot digit 0 using mean values of features
plt.title('Digit zero using mean values', fontsize=18)

"""##Βήμα 8"""

plt.imshow(np.reshape(zero_variance_values,(16,16)),cmap='gray')   # plot digit 0 using variance values of features
plt.title('Digit zero using variance values', fontsize=18)

"""##Βήμα 9"""

mean_values=np.zeros((10,256))                                               # table that will have mean values for all features of digit i in row i
variance_values=np.zeros((10,256))                                           # table that will have variance values for all features of digit i in row i

for digit in range (0,10):
  mean_values[digit,:]=lib.digit_mean(X_train,y_train,digit)
  variance_values[digit,:]=lib.digit_variance(X_train,y_train,digit)

fig=plt.figure(figsize=(22,10))
fig.suptitle('Digits using mean values', fontsize=22)
for i in range(0,10):
  ax=fig.add_subplot(2,5,i+1)
  ax.imshow(np.reshape(mean_values[i],(16,16)),cmap='gray')                 # plot each digit using the mean values for all its features
  plt.title('Digit {}'.format(i), fontsize=18)

"""##Bήμα 10"""

prediction=lib.euclidean_distance_classifier(X_test[100,:],mean_values)   # get prediction of euclidean classiffier for test sample with index 101, regarding which digit it is
print("Prediction class is:",prediction)
print("Actual class is:", int(y_test[100]))
if (prediction==int(y_test[100])):
  print("Prediction is correct.")
else:
  print("Prediction is incorrect.")

"""##Bήμα 11

α)
"""

predictions=lib.euclidean_distance_classifier(X_test,mean_values)  # get predictions of euclidean classiffier () for test data, predictions table holds at index i the prediction about which class sample i belongs to

"""β)"""

percentage=np.sum(predictions==y_test)/len(y_test)            # calculate percentage of correct predictions for test data
print("Success rate is:", percentage)

"""##Βήμα 12"""

classifier=lib.EuclideanDistanceClassifier()                  # initialize an instance of a scikit-learn euclidean distance classifier
classifier.fit(X_train,y_train)                               # train classifier
print("Success rate is:",classifier.score(X_test,y_test))     # print success rate of classifier for test data

"""##Βήμα 13

α)
"""

score=lib.evaluate_classifier(lib.EuclideanDistanceClassifier(), X_train, y_train)        # calculate mean score of k-fold cross-validation on all data

print('Mean score with 5-fold cross-validation is:', score)

"""β)"""

pca = PCA(n_components=2)                                                                                                     # in order to plot the desicion regions we will use PCA to reduce dimensionality (256 features --> 2 features)
pca_test  = pca.fit_transform(X_train)                                                                                        # adapt data to reduced demensionality keeping 2 principal components (2 features)
print("First component contains {:.2f}% of the total information.".format((100*pca.explained_variance_ratio_[0])))
print("Second component contains {:.2f}% of the total information.".format( (100*pca.explained_variance_ratio_[1])))
print("The 2 components combined contain {:.2f}% of the total information.".format((100*pca.explained_variance_ratio_[0])+(100*pca.explained_variance_ratio_[1])) )

def plot_clf(clf, X, y, with_samples=True):                                                                                  # function that plots decision surface of classifier clf, for feature table X and label table y

    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1)
    X0, X1 = X[:, 0], X[:, 1]                                                                                                # X0 will keep values of the one principal component and X1 of the other
    classes = sorted(set(y))                                                                                                 # get distinct labels of classes
    colors = ['peru', 'maroon', 'silver', 'tomato', 'lightgreen', 'orange', 'bisque', 'forestgreen', 'violet', 'lightpink']  # each color will represent a different digit

    x_min, x_max = X0.min() - 1, X0.max() + 1                                                                                # min and max value for x axis
    y_min, y_max = X1.min() - 1, X1.max() + 1                                                                                # min and max value for y axis
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.05))                                        # return coordinate matrices from coordinate vectors of grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8,levels=11)                                                  # draw contours for decision regions, levels: determines the number and positions of the contour lines / regions.

    if (with_samples):
      for i in range (0,len(classes)):                                                                                       # draw scatter plot for the 2 chosen features by PCA, for all the samples of each digit
        ax.scatter(X0[y==i],X1[y==i],c=colors[i],label=int(classes[i]),s=60,alpha=0.9,edgecolors='k')

    ax.set_ylabel('PCA component 2')
    ax.set_xlabel('PCA component 1')
    if (with_samples):
      ax.set_title('Decision surface of Euclidean Distance Classifier and samples', fontsize=18)
    else:
      ax.set_title('Decision surface of Euclidean Distance Classifier', fontsize=18)
    ax.set_xticks(())
    ax.set_yticks(())
    if (with_samples):
      ax.legend()
    else:
      col = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in out.collections]  #create proxy artists to make the legend
      ax.legend(col,{int(i) for i in classes})
    plt.show()

clf = lib.EuclideanDistanceClassifier()           # initialize an euclidean distance classifier
pca = PCA(n_components=2)                         # set a PCA with 2 principal components
pca_train  = pca.fit_transform(X_train)           # adapt data to reduced demensionality keeping 2 principal components (2 features)
clf.fit(pca_train, y_train)                       # train classifier for train data
plot_clf(clf, pca_train, y_train)                 # call function to plot desicion surface and train data

plot_clf(clf, pca_train, y_train, False)           # call function to plot desicion surface without plotting train data

"""γ)"""

train_sizes, train_scores, test_scores =learning_curve(lib.EuclideanDistanceClassifier(),X_train, y_train, cv=5, n_jobs=-1, shuffle=True)   # calculate learning curve for euclidean distance classifier using train data, set cross validation to 5 fold, n_jobs set to use all processors, shuffle=True for randomness in choice of training data

print("Train sizes are: ")
print(train_sizes)
print("Train scores are: ")
print(train_scores)
print("Test scores are:")
print(test_scores)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Euclidean Distance Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""## Βήμα 14"""

priors=lib.calculate_priors(X_train,y_train)            # calculate the apriori probability of each digit

for i in range(10):
  print("The apriori probability of class " + str(i) + " is : "+ str(round(priors[i]*100,3)) +"%")

fig = plt.figure()
digits=np.arange(10)
plt.bar(digits, priors)
plt.xticks(digits,digits)
plt.title('Apriori Probability of Digits')
plt.xlabel('Digit')
plt.ylabel('Apriori Probability')
plt.show()

"""## Βήμα 15

α)
"""

param_grid = {'smoothing': [0.00001, 0.00005, 0.0001, 0.0003, 0.0005,0.0008]}                   # options for smoothing parameter for Naive Bayes classifier
grid = GridSearchCV(lib.CustomNBClassifier(), param_grid, refit = True,  cv = 5, n_jobs = -1)   # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                      # run fit with all sets of parameters

print("Best parameter for Naive Bayes Classifier is : " +str(grid.best_params_))                # print best set of parameters

classifier=lib.CustomNBClassifier(0.0008)               # create Naive Bayes Classifier instance
classifier.fit(X_train,y_train)                   # train classifier
predictions=classifier.predict(X_test)            # get predictions for test set
print("Predicted labels are: ",predictions)

"""β)"""

score=classifier.score(X_test,y_test)             # calculate accuracy of classifier
print("Custom Naive Bayes Classifier has an accuracy of: ", score)
print("Custom Naive Bayes Classifier with 5 cross-validation has a mean accuracy of: ",lib.evaluate_custom_nb_classifier(X_train,y_train))

"""γ)"""

param_grid = {'var_smoothing': [0.00001, 0.00005, 0.0001, 0.0003, 0.0005,0.0008]}               # options for smoothing parameter for Naive Bayes classifier
grid = GridSearchCV(GaussianNB(), param_grid, refit = True,  cv = 5, n_jobs = -1)               # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                      # run fit with all sets of parameters

print("Best parameter for Naive Bayes Classifier is : " +str(grid.best_params_))                # print best set of parameters

clf = GaussianNB(var_smoothing=0.0008)           # Create scikit-learn Naive Bayes classifier instance
clf.fit(X_train, y_train)                         # train classifier
score=clf.score(X_test,y_test)                    # calculate accuracy of classifier
print("Scikit-learn Naive Bayes Classifier has an accuracy of: ", score)
print("Scikit-learn Naive Bayes Classifier with 5 cross validation has an accuracy of: ", lib.evaluate_sklearn_nb_classifier(X_train,y_train))

"""##Βήμα 16

α)
"""

param_grid = {'smoothing': [0.00001, 0.00005, 0.0001, 0.0003, 0.0005,0.0008], 'use_unit_variance' : [True]}     # options for smoothing parameter for Naive Bayes classifier
grid = GridSearchCV(lib.CustomNBClassifier(True), param_grid, refit = True,  cv = 5, n_jobs = -1)               # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                                      # run fit with all sets of parameters

print("Best parameter for Naive Bayes Classifier is : " +str(grid.best_params_))                                # print best set of parameters

classifier=lib.CustomNBClassifier(True, smoothing=0.00001)     # create Naive Bayes Classifier instance, with all variances equal to one
classifier.fit(X_train,y_train)                                # train classifier
predictions=classifier.predict(X_test)                         # get predictions for test set
print("Predicted labels are: ",predictions)

"""β)"""

score=classifier.score(X_test,y_test)             # calculate accuracy of classifier
print("Naive Bayes Classifier with unit variances has an accuracy of: ", score)
print("Naive Bayes Classifier with unit variances has an  5-fold accuracy of: ",lib.evaluate_classifier(classifier,X_train,y_train,5))

"""##Βήμα 17

### SVM Classifier

####Linear SVM Classifier
"""

param_grid = {'C': [0.01, 0.1, 1, 10,20],'gamma':['scale','auto'],'kernel': ['linear']}   # options for parameters for linear SVM classifier
grid = GridSearchCV(svm.SVC(), param_grid, refit = True,  cv = 5, n_jobs = -1)            # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                # run fit with all sets of parameters

print("Best parameters for linear SVM Classifier are : " +str(grid.best_params_))         # print best set of parameters

#Evaluation of linear SVM model
score=lib.evaluate_linear_svm_classifier(X_train,y_train)
print('Mean score of linear SVM classifier with 5-fold cross-validation is: ',score)

train_sizes, train_scores, test_scores = learning_curve(svm.SVC(C= 0.01, gamma= 'scale', kernel= 'linear'), X_train,y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Best Linear SVM Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""####RBF SVM Classifier"""

param_grid = {'C': [0.01, 0.1, 1, 10,20],'gamma':['scale','auto'],'kernel': ['rbf']} # options for parameters for rbf SVM classifier
grid = GridSearchCV(svm.SVC(), param_grid, refit = True,  cv = 5, n_jobs = -1)       # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                            # run fit with all sets of parameters

print("Best parameters for linear SVM Classifier are : " +str(grid.best_params_))    # print best set of parameters

#Evaluation of rbf SVM model
score=lib.evaluate_rbf_svm_classifier(X_train,y_train)
print('Mean score of rbf SVM classifier with 5-fold cross-validation is: ',score)

train_sizes, train_scores, test_scores = learning_curve(svm.SVC(C= 10, gamma= 'scale', kernel= 'rbf'), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Best RBF SVM Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""####Best SVM Classifier"""

param_grid = {'C': [0.01, 0.1, 1, 10],'gamma':['scale','auto'],'kernel': ['linear','rbf', 'poly','sigmoid']}  # options for parameters for  SVM classifier

grid = GridSearchCV(svm.SVC(), param_grid, refit = True,  cv = 5, n_jobs = -1)            # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                # run fit with all sets of parameters

print("Best parameters for SVM model are : " +str(grid.best_params_))                    # print best set of parameters

#Evaluation of SVM model
score=lib.evaluate_svm_classifier(X_train,y_train)    #####ΓΙΑΤΙ ΒΓΑΙΝΕΙ ΧΑΜΗΛΟΤΕΡΟ ΑΠΟ ΤΟ ΑΠΟ ΠΑΝΩ
print('Mean score of polynomial SVM classifier with 5-fold cross-validation is: ',score)

train_sizes, train_scores, test_scores = learning_curve(svm.SVC(C= 10, gamma= 'auto', kernel= 'poly'), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Polynomial SVM Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""### Nearest Neighbors Classifier"""

param_grid = {'n_neighbors':[2,3,4,5,6,7,8], 'weights': ['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree']}   # options for parameters for KNN classifier
grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True,  cv = 5, n_jobs = -1)   # initialize GridSearchCV instance
grid.fit(X_train, y_train)                                                                    # run fit with all sets of parameters

print("Best parameters for Nearest Neighbors Classifier are : " +str(grid.best_params_))      # print best set of parameters

score=lib.evaluate_knn_classifier(X_train,y_train)            # evaluate Nearest Neighbors classifier (using best set of parameters)
print("Mean score of KNN classifier with 5-fold cross-validation is: ",score)

train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors=2,weights='distance',algorithm='auto'), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("KNN Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""###Naive Bayes

####Custom
"""

score=lib.evaluate_custom_nb_classifier(X_train,y_train)
print("Mean score of Custom Naive Bayes Classifier with 5 cross-validation: ",score)

train_sizes, train_scores, test_scores = learning_curve(lib.CustomNBClassifier(smoothing=0.0008), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Custom Naive Bayes Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""####Scikit-learn implementation"""

score=lib.evaluate_sklearn_nb_classifier(X_train,y_train)
print("Mean score of scikit-learn Naive Bayes Classifier with 5 cross-validation: ",score)

train_sizes, train_scores, test_scores = learning_curve(GaussianNB(var_smoothing=0.008), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Scikit-learn Naive Bayes Classifier learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""####Custom with variances=1"""

score=lib.evaluate_custom_nb_classifier(X_train,y_train,folds=5,use_unit_variance=True, smoothing=0.00001)
print("Mean score of Custom Naive Bayes Classifier with 5 cross-validation: ",score)

train_sizes, train_scores, test_scores = learning_curve(lib.CustomNBClassifier(use_unit_variance=True,smoothing=0.00001), X_train, y_train, cv = 5,n_jobs = -1,shuffle=True)

train_scores_mean = np.mean(train_scores, axis=1)                                # compute mean value of scores for train set
train_scores_std = np.std(train_scores, axis=1)                                  # compute standard deviation for train set
test_scores_mean = np.mean(test_scores, axis=1)                                  # compute mean value of scores for test set
test_scores_std = np.std(test_scores, axis=1)                                    # compute standard deviation for train set

plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")   # fill area between mean+std, mean-std
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1,color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")
plt.title("Custom Naive Bayes Classifier with unit variances learning curve", fontsize=18)
plt.ylabel("Score")
plt.xlabel("Number of training samples")

"""##Βήμα 18

###α) Voting Classifier
"""

# initialize and train classifiers
euclidean=lib.EuclideanDistanceClassifier()
euclidean.fit(X_train,y_train)
custom_naive_bayes=lib.CustomNBClassifier(smoothing=0.0008)
custom_naive_bayes.fit(X_train,y_train)
custom_naive_bayes_unit=lib.CustomNBClassifier(True,smoothing=0.0008)
custom_naive_bayes_unit.fit(X_train,y_train)
naive_bayes=GaussianNB(var_smoothing=0.008)
naive_bayes.fit(X_train,y_train)
svm_best=svm.SVC(C= 10, gamma= 'auto', kernel= 'poly', probability=True)   # probability set to true to be used at soft voting
svm_best.fit(X_train,y_train)
svm_rbf=svm.SVC(C= 10, gamma= 'scale', kernel= 'rbf', probability=True)
svm_rbf.fit(X_train,y_train)
svm_linear=svm.SVC(C= 0.01, gamma= 'scale', kernel= 'linear', probability=True)
svm_linear.fit(X_train,y_train)
knn=KNeighborsClassifier(n_neighbors=2,weights='distance',algorithm='auto')
knn.fit(X_train,y_train)

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

"""####Confusion matrices"""

matrix = confusion_matrix(y_train, euclidean.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for Euclidean classifier")

matrix = confusion_matrix(y_train, custom_naive_bayes.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for Custom Naive Bayes classifier")

matrix = confusion_matrix(y_train, custom_naive_bayes_unit.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for Custom Naive Bayes classifier with unit variances")

matrix = confusion_matrix(y_train, naive_bayes.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for scikit-learn Naive Bayes classifier")

matrix = confusion_matrix(y_train, svm_linear.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for Linear SVM classifier")

matrix = confusion_matrix(y_train, svm_rbf.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for RBF SVM classifier")

matrix = confusion_matrix(y_train, svm_best.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for Polynomial SVM classifier")

matrix = confusion_matrix(y_train, knn.predict(X_train))
plot_confusion_matrix(matrix, list(set(y_train)), title="Confusion matrix for KNN classifier")

"""####Hard Voting Classifier with classifiers that make different errors"""

hard_voting_classifier = VotingClassifier(estimators=[('Νaive bayes',naive_bayes), ('SVM', svm_best), ('KNN', knn)], voting = 'hard')  # initialize voting classifier with hard voting
hard_voting_classifier.fit(X_train, y_train)                        # train classifier
predictions_hard = hard_voting_classifier.predict(X_test)           # get predictions of classifier for test data

score=np.sum(y_test == predictions_hard) / y_test.shape[0]
print('Accuracy score of hard voting classifier is:', score)

score=lib.evaluate_voting_classifier(X_test,y_test,estimators=[('Custom naive bayes',naive_bayes), ('SVM', svm_best), ('KNN', knn)],folds=5,hard_or_soft='hard')
print('Mean score of hard voting classifier with 5-fold cross-validation is: ', score)

"""####Hard Voting Classifier with classifiers that make similar errors"""

hard_voting_classifier = VotingClassifier(estimators=[('SVM linear',svm_linear), ('SVM', svm_best), ('SVM rbf', svm_rbf)], voting = 'hard')  # initialize voting classifier with hard voting
hard_voting_classifier.fit(X_train, y_train)                        # train classifier
predictions_hard = hard_voting_classifier.predict(X_test)           # get predictions of classifier for test data

score=np.sum(y_test == predictions_hard) / y_test.shape[0]
print('Accuracy score of hard voting classifier is:', score)

score=lib.evaluate_voting_classifier(X_test,y_test,estimators=[('SVM linear',svm_linear), ('SVM', svm_best), ('SVM rbf', svm_rbf)],folds=5,hard_or_soft='hard')
print('Mean score of hard voting classifier with 5-fold cross-validation is: ', score)

"""####Soft Voting Classifier with classifiers that make different errors"""

soft_voting_classifier = VotingClassifier(estimators=[('naive bayes',naive_bayes), ('SVM', svm_best), ('KNN', knn)], voting = 'soft')  # initialize voting classifier with hard voting
soft_voting_classifier.fit(X_train, y_train)                        # train classifier
predictions_soft = soft_voting_classifier.predict(X_test)           # get predictions of classifier for test data

score=np.sum(y_test == predictions_soft) / y_test.shape[0]
print('Accuracy score of soft voting classifier is:', score)

score=lib.evaluate_voting_classifier(X_test,y_test,estimators=[('naive bayes',naive_bayes), ('SVM', svm_best), ('KNN', knn)],folds=5,hard_or_soft='soft')
print('Mean score of soft voting classifier with 5-fold cross-validation is: ', score)

"""####Soft Voting Classifier with classifiers that make similar errors"""

soft_voting_classifier = VotingClassifier(estimators=[('SVM linear',svm_linear), ('SVM', svm_best), ('SVM rbf', svm_rbf)], voting = 'soft')  # initialize voting classifier with hard voting
soft_voting_classifier.fit(X_train, y_train)                        # train classifier
predictions_soft = soft_voting_classifier.predict(X_test)           # get predictions of classifier for test data

score=np.sum(y_test == predictions_soft) / y_test.shape[0]
print('Accuracy score of soft voting classifier is:', score)

score=lib.evaluate_voting_classifier(X_test,y_test,estimators=[('SVM linear',svm_linear), ('SVM', svm_best), ('SVM rbf', svm_rbf)],folds=5,hard_or_soft='soft')
print('Mean score of soft voting classifier with 5-fold cross-validation is: ', score)

"""###β) Bagging Classifier

#### SVM
"""

bagging_classifier=BaggingClassifier(base_estimator=svm_best,n_jobs=-1)
bagging_classifier.fit(X_train,y_train)                                       #train classifier
bag_predictions=bagging_classifier.predict(X_test)                            #get predictions of classifier for test data

score=np.sum(y_test == bag_predictions) / y_test.shape[0]
print('Accuracy score of SVM Bagging Classifier is:', score)

score=lib.evaluate_bagging_classifier(X_test,y_test,estimators=svm_best,folds=5)
print('Mean score of svm Bagging Classifier with 5-fold cross-validation is: ', score)

"""#### Decision Tree Bagging"""

decision_bagging = BaggingClassifier(n_jobs = -1)
decision_bagging.fit(X_train, y_train)
decision_pred = decision_bagging.predict(X_test)

score=np.sum(y_test == decision_pred) / y_test.shape[0]
print('Accuracy score of Decision Tree Bagging Classifier is:', score)

score=lib.evaluate_bagging_classifier(X_test,y_test,estimators=None,folds=5)
print('Mean score of Decision Tree Bagging Classifier with 5-fold cross-validation is: ', score)

"""# Βήμα 19

####α)
"""

class DigitDataset():
  def __init__(self, filename):
    self.data = pd.read_csv(filename, sep = " ", header = None)               # read file with name "filename"

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample=self.data.iloc[idx]                                                # get row with index idx
    label=sample[0]                                                           # get first value of row (label)
    features=torch.tensor(sample[1:257].values)                               # get all features of row and convert to tensor
    return(features, label)

batch_size=32
# train_dataset = DigitDataset('/content/drive/My Drive/Colab Notebooks/pattern_rec/data/train.txt')   # create train data DigitDataset instance
train_dataset = DigitDataset('/data/train.txt')   # create train data DigitDataset instance

# test_dataset = DigitDataset('/content/drive/My Drive/Colab Notebooks/pattern_rec/data/test.txt')    # create test data DigitDataset instance
test_dataset = DigitDataset('/data/test.txt')    # create test data DigitDataset instance

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)                # create train data Dataloader
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)                  # create test date Dataloader

"""####β)

#####2-Layer NN
"""

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out,activation='RELU'):
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        if (activation=='RELU'):
          self.activation = torch.nn.ReLU()
        else:
          self.activation = torch.nn.Sigmoid()

    def forward(self, x):

        #x = x.view(x.size(0), -1)

        x =  self.activation(self.linear1(x))

        return self.linear2(x)

"""Implement two layer NNs with different number of neurons and different activation functions:"""

EPOCHS = 40
IN, H , OUT = 256, 128, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

two_layer_net_RELU_1 = TwoLayerNet(IN, H, OUT)    # create NN with two layers, activation function: ReLU

EPOCHS = 40
IN, H, OUT = 256, 32, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

two_layer_net_RELU_2 = TwoLayerNet(IN, H, OUT)    # create NN with three layers, activation function: ReLU

EPOCHS = 40
IN, H, OUT = 256, 128, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

two_layer_net_SIGMOID_1 = TwoLayerNet(IN, H, OUT, 'SIGMOID')    # create NN with two layers, activation function: sigmoid

EPOCHS = 40
IN, H, OUT = 256, 32, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

two_layer_net_SIGMOID_2 = TwoLayerNet(IN, H, OUT, 'SIGMOID')    # create NN with two layers, activation function: sigmoid

"""#####3-Layer NN"""

class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1 ,H2, D_out,activation='RELU'):
        super(ThreeLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
        if (activation=='RELU'):
          self.activation = torch.nn.ReLU()
        else:
          self.activation = torch.nn.Sigmoid()

    def forward(self, x):

        #r1 = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(r1))
        hidden_1 =  self.activation(self.linear1(x))
        x = self.activation(self.linear2(hidden_1))

        # output so no dropout here
        #return F.log_softmax(self.hidden3(x))
        #return F.relu(self.hidden3(x))
        return self.linear3(x)

"""Implement three layer NNs with different number of neurons and different activation functions:"""

EPOCHS = 40
IN, H1, H2 , OUT = 256, 128, 64, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

three_layer_net_RELU_1 = ThreeLayerNet(IN, H1, H2, OUT)    # create NN with three layers, activation function: ReLU

EPOCHS = 40
IN, H1, H2 , OUT = 256, 64, 32, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

three_layer_net_RELU_2 = ThreeLayerNet(IN, H1, H2, OUT)    # create NN with three layers, activation function: ReLU

EPOCHS = 40
IN, H1, H2 , OUT = 256, 128, 64, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

three_layer_net_SIGMOID_1 = ThreeLayerNet(IN, H1, H2, OUT, 'SIGMOID')    # create NN with three layers, activation function: sigmoid

EPOCHS = 40
IN, H1, H2 , OUT = 256, 64, 32, 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

three_layer_net_SIGMOID_2 = ThreeLayerNet(IN, H1, H2, OUT, 'SIGMOID')    # create NN with three layers, activation function: sigmoid



"""####γ)

#####Create instances of PytorchNNModels and train them:
"""

optimizer = torch.optim.SGD(two_layer_net_RELU_1.parameters(), lr = LEARNING_RATE,momentum = 0.9)                      # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                # choose loss function
two_layer_model_RELU_1=lib.PytorchNNModel(two_layer_net_RELU_1,criterion,optimizer,EPOCHS,BATCH_SIZE)    # create PytorchNNModel instance
two_layer_model_RELU_1.fit(X_train,y_train)                                                                            # train NN

optimizer = torch.optim.SGD(two_layer_net_RELU_2.parameters(), lr = LEARNING_RATE,momentum = 0.9)                      # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                # choose loss function
two_layer_model_RELU_2=lib.PytorchNNModel(two_layer_net_RELU_2,criterion,optimizer,EPOCHS,BATCH_SIZE)    # create PytorchNNModel instance
two_layer_model_RELU_2.fit(X_train,y_train)                                                                            # train NN

optimizer = torch.optim.SGD(two_layer_net_SIGMOID_1.parameters(), lr = LEARNING_RATE,momentum = 0.9)                         # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                      # choose loss function
two_layer_model_SIGMOID_1=lib.PytorchNNModel(two_layer_net_SIGMOID_1,criterion,optimizer,EPOCHS,BATCH_SIZE)    # create PytorchNNModel instance
two_layer_model_SIGMOID_1.fit(X_train,y_train)                                                                               # train NN

optimizer = torch.optim.SGD(two_layer_net_SIGMOID_2.parameters(), lr = LEARNING_RATE,momentum = 0.9)                         # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                      # choose loss function
two_layer_model_SIGMOID_2=lib.PytorchNNModel(two_layer_net_SIGMOID_2,criterion,optimizer,EPOCHS,BATCH_SIZE)    # create PytorchNNModel instance
two_layer_model_SIGMOID_2.fit(X_train,y_train)                                                                               # train NN

optimizer = torch.optim.SGD(three_layer_net_RELU_1.parameters(), lr = LEARNING_RATE,momentum = 0.9)                         # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                     # choose loss function
three_layer_model_RELU_1=lib.PytorchNNModel(three_layer_net_RELU_1,criterion,optimizer,EPOCHS,BATCH_SIZE)     # create PytorchNNModel instance
three_layer_model_RELU_1.fit(X_train,y_train)                                                                               # train NN

optimizer = torch.optim.SGD(three_layer_net_RELU_2.parameters(), lr = LEARNING_RATE,momentum = 0.9)                         # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                     # choose loss function
three_layer_model_RELU_2=lib.PytorchNNModel(three_layer_net_RELU_2,criterion,optimizer,EPOCHS,BATCH_SIZE)     # create PytorchNNModel instance
three_layer_model_RELU_2.fit(X_train,y_train)                                                                               # train NN

optimizer = torch.optim.SGD(three_layer_net_SIGMOID_1.parameters(), lr = LEARNING_RATE,momentum = 0.9)                            # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                           # choose loss function
three_layer_model_SIGMOID_1=lib.PytorchNNModel(three_layer_net_SIGMOID_1,criterion,optimizer,EPOCHS,BATCH_SIZE)     # create PytorchNNModel instance
three_layer_model_SIGMOID_1.fit(X_train,y_train)                                                                                  # train NN

optimizer = torch.optim.SGD(three_layer_net_SIGMOID_2.parameters(), lr = LEARNING_RATE,momentum = 0.9)                            # choose optimizer
criterion = torch.nn.CrossEntropyLoss()                                                                                           # choose loss function
three_layer_model_SIGMOID_2=lib.PytorchNNModel(three_layer_net_SIGMOID_2,criterion,optimizer,EPOCHS,BATCH_SIZE)     # create PytorchNNModel instance
three_layer_model_SIGMOID_2.fit(X_train,y_train)                                                                                  # train NN

"""#####Evaluate NNs on train and validation data:"""

val_dataloader = DataLoader(two_layer_model_RELU_1.validation_dataset, batch_size =1, shuffle = False)  # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(two_layer_model_RELU_1.validation_dataset),256))                                                     # table of features
y_val=np.zeros((len(two_layer_model_RELU_1.validation_dataset)))                                                         # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                    # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=two_layer_model_RELU_1.score(X_train,y_train)
val_score=two_layer_model_RELU_1.score(X_val,y_val)
print('Score on train data for two layer NN: ', training_score)
print('Score on validation data for two layer NN: ', val_score)

val_dataloader = DataLoader(two_layer_model_RELU_2.validation_dataset, batch_size =1, shuffle = False)  # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(two_layer_model_RELU_2.validation_dataset),256))                                                     # table of features
y_val=np.zeros((len(two_layer_model_RELU_2.validation_dataset)))                                                         # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                    # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=two_layer_model_RELU_2.score(X_train,y_train)
val_score=two_layer_model_RELU_2.score(X_val,y_val)
print('Score on train data for two layer NN: ', training_score)
print('Score on validation data for two layer NN: ', val_score)

val_dataloader = DataLoader(two_layer_model_SIGMOID_1.validation_dataset, batch_size =1, shuffle = False)  # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(two_layer_model_SIGMOID_1.validation_dataset),256))                                                        # table of features
y_val=np.zeros((len(two_layer_model_SIGMOID_1.validation_dataset)))                                                            # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                       # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=two_layer_model_SIGMOID_1.score(X_train,y_train)
val_score=two_layer_model_SIGMOID_1.score(X_val,y_val)
print('Score on train data for two layer NN: ', training_score)
print('Score on validation data for two layer NN: ', val_score)

val_dataloader = DataLoader(two_layer_model_SIGMOID_2.validation_dataset, batch_size =1, shuffle = False)  # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(two_layer_model_SIGMOID_2.validation_dataset),256))                                                        # table of features
y_val=np.zeros((len(two_layer_model_SIGMOID_2.validation_dataset)))                                                            # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                       # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=two_layer_model_SIGMOID_2.score(X_train,y_train)
val_score=two_layer_model_SIGMOID_2.score(X_val,y_val)
print('Score on train data for two layer NN: ', training_score)
print('Score on validation data for two layer NN: ', val_score)

val_dataloader = DataLoader(three_layer_model_RELU_1.validation_dataset, batch_size =1, shuffle = False)   # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(three_layer_model_RELU_1.validation_dataset),256))                                                        # table of features
y_val=np.zeros((len(three_layer_model_RELU_1.validation_dataset)))                                                            # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                       # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=three_layer_model_RELU_1.score(X_train,y_train)
val_score=three_layer_model_RELU_1.score(X_val,y_val)
print('Score on train data for three layer NN: ', training_score)
print('Score on validation data for three layer NN: ', val_score)

val_dataloader = DataLoader(three_layer_model_RELU_2.validation_dataset, batch_size =1, shuffle = False)   # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(three_layer_model_RELU_2.validation_dataset),256))                                                        # table of features
y_val=np.zeros((len(three_layer_model_RELU_2.validation_dataset)))                                                            # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                       # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=three_layer_model_RELU_2.score(X_train,y_train)
val_score=three_layer_model_RELU_2.score(X_val,y_val)
print('Score on train data for three layer NN: ', training_score)
print('Score on validation data for three layer NN: ', val_score)

val_dataloader = DataLoader(three_layer_model_SIGMOID_1.validation_dataset, batch_size =1, shuffle = False)   # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(three_layer_model_SIGMOID_1.validation_dataset),256))                                                           # table of features
y_val=np.zeros((len(three_layer_model_SIGMOID_1.validation_dataset)))                                                               # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                          # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=three_layer_model_SIGMOID_1.score(X_train,y_train)
val_score=three_layer_model_SIGMOID_1.score(X_val,y_val)
print('Score on train data for three layer NN: ', training_score)
print('Score on validation data for three layer NN: ', val_score)

val_dataloader = DataLoader(three_layer_model_SIGMOID_2.validation_dataset, batch_size =1, shuffle = False)   # create validation data Dataloader from validation set of NN
X_val=np.zeros((len(three_layer_model_SIGMOID_2.validation_dataset),256))                                                           # table of features
y_val=np.zeros((len(three_layer_model_SIGMOID_2.validation_dataset)))                                                               # table of labels
for i,(features,label) in enumerate(val_dataloader):                                                          # get data from valudation Dataloader into numpy array
  X_val[i,:]=np.array(features)
  y_val[i]=float(label)

training_score=three_layer_model_SIGMOID_2.score(X_train,y_train)
val_score=three_layer_model_SIGMOID_2.score(X_val,y_val)
print('Score on train data for three layer NN: ', training_score)
print('Score on validation data for three layer NN: ', val_score)

"""####δ)

Evaluate on test data:
"""

test_score=two_layer_model_RELU_1.score(X_test,y_test)
print('Score on test data for first two layer NN with ReLU activation function: ', test_score)
test_score=two_layer_model_RELU_2.score(X_test,y_test)
print('Score on test data for second two layer NN with ReLU activation function: ', test_score)
test_score=two_layer_model_SIGMOID_1.score(X_test,y_test)
print('Score on test data for first two layer NN with sigmoid activation function: ', test_score)
test_score=two_layer_model_SIGMOID_2.score(X_test,y_test)
print('Score on test data for second two layer NN with sigmoid activation function: ', test_score)
test_score=three_layer_model_RELU_1.score(X_test,y_test)
print('Score on test data for first three layer NN with ReLU activation function: ', test_score)
test_score=three_layer_model_RELU_2.score(X_test,y_test)
print('Score on test data for second three layer NN with ReLU activation function: ', test_score)
test_score=three_layer_model_SIGMOID_1.score(X_test,y_test)
print('Score on test data for first three layer NN with sigmoid activation function: ', test_score)
test_score=three_layer_model_SIGMOID_2.score(X_test,y_test)
print('Score on test data for second three layer NN with sigmoid activation function: ', test_score)
