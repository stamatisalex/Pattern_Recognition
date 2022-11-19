from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import random
import statistics
import sys
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,BaggingClassifier
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_digit_indexes(y,digit):
    '''Takes a table of labels returns all indexes of the table where the label corresponds to a specific digit/class.

    Args:
        digit (int): The digit whose indexes we are searching.
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        indexes (np.ndarray): A 1d table with the indexes found.
    '''
    indexes=[]
    for index,label in enumerate(y):        #loop through labels
        val=int(label)
        if val==digit:                      # if the label corresponds to the digit
            indexes.append(index)           # add the index to the table
    return indexes


def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    digit=np.reshape(X[index,:],(int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1]))))  # trasform 1d into a square 2d array
    plt.imshow(digit,cmap='gray')                                                     # show digit
    plt.title('Digit with index {}'.format(index+1), fontsize=18)


def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    indexes_arr=[]                                      # 2d array where row i will contain the indexes that correspond to the rows of the data that are a sample of digit i
    num_of_classes=len(set(y))                          # get number of discrete classes from the labels
    for i in range (0,num_of_classes):                  # for all classes (digits)
        indexes_arr.append(get_digit_indexes(y,i))      # get indexes
    fig=plt.figure(figsize=(22,10))                     # plot a random sample from each class:
    fig.suptitle('Randomly picked sample picture for each class', fontsize=22)
    for i in range(0,num_of_classes):
        ax=fig.add_subplot(2,5,i+1)
        ax.imshow(np.reshape(X[random.choice(indexes_arr[i]),:],(int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))),cmap='gray')  # a random index is chosen for each class
        plt.title('Digit {}'.format(i), fontsize=18)

def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    index=(pixel[0]-1)*int(np.sqrt(X.shape[1]))+pixel[1]    # get which of all the features the specific pixel corresponds to
    indexes=get_digit_indexes(y,digit)                      # get all indexes for the specific digit in the data
    return statistics.mean(X[indexes,index-1])              # return mean value for the specific pixel/feature




def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    index=(pixel[0]-1)*int(np.sqrt(X.shape[1]))+pixel[1]   # get which of all the features the specific pixel corresponds to
    indexes=get_digit_indexes(y,digit)                     # get all indexes for the specific digit in the data
    return statistics.variance(X[indexes,index-1])         # return variance for the specific pixel/feature



def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    mean_values=np.zeros(X.shape[1])                       # table where the mean for all instances of a specific digit will be kept
    for i in range (0,int(np.sqrt(X.shape[1]))):           # calculate mean value for all features
        for j in range (0,int(np.sqrt(X.shape[1]))):
            mean_values[i*int(np.sqrt(X.shape[1]))+j]=digit_mean_at_pixel(X,y,digit,(i+1,j+1))
    index=0
    for i in range (int(np.sqrt(X.shape[1]))**2,X.shape[1]):
        mean_values[i]=digit_mean_at_pixel(X,y,digit,(int(np.sqrt(X.shape[1]))+1,index+1))
        index+=1
    return mean_values



def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    variance_values=np.zeros(X.shape[1])                # table where the variance for all instances of a specific digit will be kept
    for i in range (0,int(np.sqrt(X.shape[1]))):        # calculate variance for all features
        for j in range (0,int(np.sqrt(X.shape[1]))):
            variance_values[i*int(np.sqrt(X.shape[1]))+j]=digit_variance_at_pixel(X,y,digit,(i+1,j+1))
    return variance_values


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    return np.sqrt(np.sum((s-m)**2))    # return euclidean distance of the two vectors

def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    if (X.ndim==1):
        predictions=np.array(np.argmin([euclidean_distance(X,X_mean[j,:]) for j in range(0,X_mean.shape[0])]))
    else:
        predictions=np.zeros(X.shape[0])            # table that will contain the predictions
        for i in range(0,X.shape[0]):               # for each sample of data
            predictions[i]=np.argmin([euclidean_distance(X[i,:],X_mean[j,:]) for j in range(0,X_mean.shape[0])] )   #find the index of the template vector that has the minimum euclidean distance with the feature vector of this sample
    return predictions                              # predictions table contains for each sample the index of X_mean where the euclidean distance is minimum
                                                    # since we know which class each index of X_mean belongs to (digits 0 to 9 in this order), we have the final prediction


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        """
        classes=set(y)
        mean_values=np.zeros((len(classes),X.shape[1]))

        for item in classes:
            mean_values[int(item),:]=digit_mean(X,y,int(item))
        self.X_mean=mean_values

        return self
        """
        classes=set(y)                                      # get all distinct classes from the labels
        classes=sorted(classes)                             # sort the classes (digits)

        mean_values=np.zeros((len(classes),X.shape[1]))     # table that will contain the mean feature values

        for i in range(0,len(classes)):
            mean_values[i,:]=digit_mean(X,y,classes[i])
        self.X_mean_=mean_values

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        return euclidean_distance_classifier(X,self.X_mean_)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions=self.predict(X)                 # get classifier's predictions
        return np.sum(predictions==y)/len(y)        # calculate percantage of correct ones




def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring="accuracy")    # calculate k-fold cross evaluation, using default random state
    print("Scores with {}-fold cross-validation are:".format(folds), scores)
    return np.mean(scores)                                                               # return mean score



def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    samples=X.shape[0]                                         # get number of samples
    element, occurence=np.unique(y,return_counts=True)         # calculate how often each label occurs
    priors=np.zeros(len(set(y)))                               # array that will hold a-priori probability for each class
    for i,times in enumerate(occurence):
      priors[i] = times/samples                                # calculate probability for each label
    return priors



class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, smoothing=0.00001):
        self.use_unit_variance = use_unit_variance
        self.mean_=None
        self.covmatrix_=None
        self.priors_=None
        self.smoothing=smoothing


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        num_of_classes=len(set(y))                                          # number of distinct labels
        smoothing = self.smoothing                                          # smoothing parameter
        mymean = np.zeros((num_of_classes,X.shape[1]))                      # table to hold mean values for features of each class
        mycovmatrix = np.zeros((num_of_classes,X.shape[1],X.shape[1]))      # covariance table

        if (not self.use_unit_variance):
            mycov = np.zeros((num_of_classes,X.shape[1]))                   # table to hold variance values for features of each class
            for class_ in range(num_of_classes):
                mymean[class_,:] = digit_mean(X,y,class_)                   # calculate mean values of characteristics for each class
                mycov[class_,:] = digit_variance(X,y,class_)                # calculate variance values of characteristics for each class
        else:
            for class_ in range(num_of_classes):
                mymean[class_,:] = digit_mean(X,y,class_)                   # calculate mean values of characteristics for each class
            mycov = np.ones((num_of_classes,X.shape[1]))                    # set variance value equal to one for all features

        for class_ in range(num_of_classes):
            for feature in range(X.shape[1]):
                mycovmatrix[class_,feature,feature] = mycov[class_,feature] + smoothing    # diagonal elements (variances) are added and smoothing is added too, all other covariances are zero (uncorrelated features)


        self.mean_=mymean
        self.covmatrix_=mycovmatrix
        self.priors_=calculate_priors(X, y)

        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        multivariance
        """
        multipdfs = np.zeros((self.mean_.shape[0],X.shape[0]))
        for i in range(self.mean_.shape[0]):
            multipdfs[i,:] = multivariate_normal.logpdf(X, mean = self.mean_[i], cov = self.covmatrix_[i])
            multipdfs[i,:] += np.log(self.priors_[i])
        return np.argmax(multipdfs,axis = 0)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions=self.predict(X)                 # get classifier's predictions
        return np.sum(predictions==y)/len(y)        # calculate percantage of correct ones


class DigitDataset():
  def __init__(self, X, y):
    self.features=torch.from_numpy(X)               # convert numpy array to tensor
    self.labels=torch.from_numpy(y)

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    features=self.features[idx]                      # get features of sample idx
    label=self.labels[idx]                           # get label of sample idx
    return(features, label)

class DigitTestDataset():
  def __init__(self, X):
    self.features=torch.from_numpy(X)                # convert numpy array to tensor

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    features=self.features[idx]                      # get features of sample idx
    return(features)

def train_net(net,EPOCHS,train_dataloader,criterion,optimizer):    # function to train NN
  net.train()
  for epoch in range(EPOCHS):                                      # loop for all epochs
      running_average_loss = 0
      for i, (features, label) in enumerate(train_dataloader):     # loop for all batches
          optimizer.zero_grad()
          out = net(features.float())                              # forward pass
          loss = criterion(out, label.long())                      # compute per batch loss
          loss.backward()                                          # compurte gradients based on the loss function
          optimizer.step()                                         # update weights

          running_average_loss += loss.detach().item()             # compute average loss
          if i % 100 == 0:
              print("Epoch: {} \t Batch: {} \t Loss {}".format(epoch, i, float(running_average_loss) / (i + 1)))


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, criterion, optimizer,epochs=40,batch_size=32):
        # initialize model, criterion and optimizer and other variables
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs=epochs
        self.train_dataset=None
        self.validation_dataset=None
        self.batch_size=batch_size

    def fit(self, X, y):
        train_dataset = DigitDataset(X,y)                 # create train data DigitDataset instance
        train_size = int(0.7 * len(train_dataset))        # keep 70% of data for training
        validate_size = len(train_dataset) - train_size   # keep 30% of data for validation
        train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validate_size])  # split data
        train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)  # create train data Dataloader
        val_dataloader = DataLoader(validation_dataset, batch_size = self.batch_size, shuffle = True)  # create train data Dataloader
        train_net(self.model,self.epochs,train_dataloader,self.criterion,self.optimizer)
        self.validation_dataset=validation_dataset
        self.train_dataset=train_dataset
        return self


    def predict(self, X):
        # X is a numpy array where row i has the feautures of sample i
        # The function returns a 1D numpy array where item i is the predicted class for sample i

        test_dataset=DigitTestDataset(X)
        test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)

        self.model.eval()                                           # turn off batchnorm/dropout
        preds=[]                                                    # list to hold predictions
        with torch.no_grad():                                       # no gradients required
            for i, data in enumerate(test_dataloader):
                X_batch = data                                      # test data (features)
                out = self.model(X_batch.float())                   # get net's predictions
                val, y_pred = out.max(1)                            # argmax since output is a prob distribution
                preds.append(float(y_pred))                         # add prediction to prediction list

        return(np.array(preds))

    def score(self, X, y):
        # Returns accuracy score.
        predictions=self.predict(X)
        return np.sum(predictions==y)/len(y)                        # calculate percantage of correct ones

def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=svm.SVC(C=0.01,gamma='scale',kernel='linear')
    return evaluate_classifier(clf,X,y,folds)

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=svm.SVC(C=10,gamma='scale',kernel='rbf')
    return evaluate_classifier(clf,X,y,folds)

def evaluate_svm_classifier(X, y, folds=5):
    """ Create an svm and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=svm.SVC(C=10,gamma='auto',kernel='poly')
    return evaluate_classifier(clf,X,y,folds)



def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=KNeighborsClassifier(n_neighbors=2,weights='distance',algorithm='auto')
    return evaluate_classifier(clf,X,y,folds)


def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=GaussianNB(var_smoothing=0.00001)
    return evaluate_classifier(clf,X,y,folds)


def evaluate_custom_nb_classifier(X, y, folds=5, smoothing=0.0008, use_unit_variance=False):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf =CustomNBClassifier(use_unit_variance,smoothing)
    return evaluate_classifier(clf,X,y,folds)


def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=EuclideanDistanceClassifier()
    return evaluate_classifier(clf,X,y,folds)

def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError



def evaluate_voting_classifier(X, y, estimators, folds=5, hard_or_soft='hard', ):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=VotingClassifier(estimators=estimators, voting = hard_or_soft)
    return evaluate_classifier(clf,X,y,folds)



def evaluate_bagging_classifier(X, y, estimators, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf=BaggingClassifier(base_estimator=estimators,n_jobs=-1)
    return evaluate_classifier(clf,X,y,folds)
