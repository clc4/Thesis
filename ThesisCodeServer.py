# Configuring workspace

import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io
import math
import h5py
import time
import itertools
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from scipy import stats
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D

# names of all the experiments

plastics = ['wpl', 'sopl', 'sbbpl', 'rbpl', 'npl', 'lpl', 
     'cbpl', 'bpl', 'bbpl', 'apl', 'nopl']
papers = ['wpa', 'sopa', 'sbbpa', 'rbpa', 'npa', 'lpa', 
     'cbpa', 'bpa', 'bbpa', 'apa', 'nopa']
glasses = ['wg', 'sog', 'sbbg', 'rbg', 'ng', 'lg', 
     'cbg', 'bg', 'bbg', 'ag', 'nog']
boxes = ['wbx', 'sobx', 'sbbbx', 'rbbx', 'nbx', 'lbx', 
     'cbbx', 'bbx', 'bbbx', 'abx', 'nobx']

nothings = ['nopl', 'nopa', 'nog', 'nobx']

mic_compare = ['bbg_comparemic', 'wg_comparemic']

# prepares the variables given the classes

def prep(classes):

    # Organize sample data (in freq domain) where rows are different observations

    tempFile = scipy.io.loadmat('FreqSamplesConcat2/wpl30.mat')
    numFreqs = len(tempFile['data'][0])/4
    numSamples = 60
    numObser = numSamples*len(classes)
    objData_accel = np.zeros((numObser, numFreqs*3))
    objData_audio = np.zeros((numObser, numFreqs))
    
    for i in range(len(classes)):
        for j in range(numSamples):
            tempFile = scipy.io.loadmat('FreqSamplesConcat2/%s%d.mat' %(classes[i], j+1))
            data = tempFile['data'][0]
            objData_accel[i*numSamples+j, :] = data[0:3*numFreqs]
            objData_audio[i*numSamples+j, :] = data[3*numFreqs:4*numFreqs]

    # numerical label
    
    label = np.zeros((numObser, 1))
    label = np.array(label).astype(int)
    for i in range(len(classes)):
        for j in range(numSamples):
            label[i*numSamples+j] = i
            
    return objData_accel, objData_audio, label

# prepares the variables given the classes

def prepAudio(classes):

    # Organize sample data (in freq domain) where rows are different observations

    tempFile = scipy.io.loadmat('FreqSamplesConcat2/wpl30.mat')
    numFreqs = len(tempFile['data'][0])/4
    numSamples = 60
    numObser = numSamples*len(classes)
    objData_air = np.zeros((numObser, numFreqs))
    objData_contact = np.zeros((numObser, numFreqs))
    
    for i in range(len(classes)):
        for j in range(numSamples):
            tempFile = scipy.io.loadmat('FreqSamplesConcat2/%s%d.mat' %(classes[i], j+1))
            data = tempFile['data'][0]
            objData_air[i*numSamples+j, :] = data[0:numFreqs]
            objData_contact[i*numSamples+j, :] = data[numFreqs:2*numFreqs]

    # numerical label
    
    label = np.zeros((numObser, 1))
    for i in range(len(classes)):
        for j in range(numSamples):
            label[i*numSamples+j] = i
    label = label[:, 0]
            
    return objData_air, objData_contact, label

# prepares the variables given the classes, implements DFT321

def prepDFT321(classes):

    # Organize sample data (in freq domain) where rows are different observations
    # Perform DFT321 on accelerometer data, keep audio data as is
    
    tempFile = h5py.File('FreqSamples2/wpl30.mat')
    numFreqs = len(tempFile['x'])
    numSamples = 60
    numObser = numSamples*len(classes)
    objData_accel = np.zeros((numObser, numFreqs))
    objData_audio = np.zeros((numObser, numFreqs))

    for i in range(len(classes)):
        for j in range(numSamples):
            tempFile = h5py.File('FreqSamples2/%s%d.mat' %(classes[i], j+1))
            x = np.array(tempFile['x'])
            y = np.array(tempFile['y'])
            z = np.array(tempFile['z'])
            audio = np.array(tempFile['audio'])
            objData_accel[i*numSamples+j, :] = np.sqrt(x[:,0]**2 + y[:,0]**2 + z[:,0]**2)
            objData_audio[i*numSamples+j, :] = audio[:,0]
            tempFile.close()

    # numerical label
    
    label = np.zeros((numObser, 1))
    for i in range(len(classes)):
        for j in range(numSamples):
            label[i*numSamples+j] = i
    label = label[:, 0]
            
    return objData_accel, objData_audio, label

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal C parameter
# Then it tests the C parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)
# This compares results when using DFT321 versus not. 

def none_vs_DFT321(classes, numTestFolds, numTrainFolds, title, *parameters):
    
    start_time = time.time()
    
    objData_accel_none, objData_audio_none, label = prep(classes)
    objData_accel_DFT321, objData_audio_DFT321, label = prepDFT321(classes)
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
#     if len(parameters) == 1:
#         numComponents = parameters[0]
#     if len(parameters) == 2:
#         loFreq = parameters[0]
#         hiFreq = parameters[1]
#         objData = objData[:, loFreq, hiFreq]
    
    # Test this C_range
    C_range = np.logspace(-5, 2, 20)
    
    # NONE
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs = np.zeros((numTestFolds, len(C_range)))
    OptC = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold
    
    # DFT321
    # Split data into folds
    TestAccuracies_DFT321 = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies_DFT321 = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores_DFT321 = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores_DFT321 = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs_DFT321 = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs_DFT321 = np.zeros((numTestFolds, len(C_range)))
    OptC_DFT321 = np.zeros((numTestFolds, 1))
    AvgTrainTime_DFT321 = np.zeros((numTestFolds, 1)) # average training time per fold
    
    # For each fold, find an optimal C parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    
    # NONE
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData_accel_none[train,:], objData_accel_none[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(data_train_transformed2)
                    data_train_transformed2 = pca.transform(data_train_transformed2)
                    data_val_transformed2 = pca.transform(data_val_transformed2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptC]
        OptC[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        TrainScores[k,:] = train_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        TrainStderrs[k,:] = train_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    # Index for which outer fold we're in
    k = 0
    
    # DFT321
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData_accel_DFT321[train,:], objData_accel_DFT321[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(data_train_transformed2)
                    data_train_transformed2 = pca.transform(data_train_transformed2)
                    data_val_transformed2 = pca.transform(data_val_transformed2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies_DFT321[k] = val_scores[indexOptC]
        OptC_DFT321[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores_DFT321[k,:] = val_scores[:, 0]
        TrainScores_DFT321[k,:] = train_scores[:, 0]
        ValStderrs_DFT321[k,:] = val_stderrs[:, 0]
        TrainStderrs_DFT321[k,:] = train_stderrs[:, 0]
        AvgTrainTime_DFT321[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_DFT321[k]).fit(data_train_transformed, label_train)
        TestAccuracies_DFT321[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print "Reporting with no dimensionality reduction"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC.mean())
    print "Testing accuracy is %0.4f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.4f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    print "Reporting with DFT321"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime_DFT321, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC_DFT321.mean())
    print "Testing accuracy is %0.4f (+/- %0.2f)" %(TestAccuracies_DFT321.mean(), (TestAccuracies_DFT321.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.4f (+/- %0.2f)" %(ValAccuracies_DFT321.mean(), (ValStderrs_DFT321.mean())*1.96)
    
    fig1 = plt.gcf()
    axes = plt.gca()
    axes.set_ylim([50,102])
    plt.axhline(y=ValAccuracies.mean()*100, xmin=0, xmax=1, hold=None, color='g', linestyle='-.', label='Best Validation Accuracy without DFT321')
    te_a, caplines, errorlinecols = plt.errorbar(C_range, np.mean(TrainScores_DFT321, 0)*100, yerr=np.mean(TrainStderrs_DFT321, 0)*100, fmt='bo-', label='Training Accuracy with DFT321')
    tr_a, caplines, errorlinecols = plt.errorbar(C_range, np.mean(ValScores_DFT321, 0)*100, yerr=np.mean(ValStderrs_DFT321, 0)*100, fmt='ro-', label='Validation Accuracy with DFT321')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.38),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Effect of DFT321 on Accuracy: %s '%(title))
    plt.show()
    fig1.savefig('../../images/DFT321_vs_none_%s.png' %(title))
    
    return OptC.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96, OptC_DFT321.mean(), \
    TestAccuracies_DFT321.mean(), (TestAccuracies_DFT321.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies_DFT321.mean(), (ValStderrs_DFT321.mean())*1.96

def run_none_vs_DFT321(experiment, numComb, numTestFolds, numTrainFolds, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file = open("TableData/nonevsDFT321_%s.txt" % experiment, 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        optc_none, testacc_none, teststd_none, valacc_none, valstd_none, \
        optc_dft, testacc_dft, teststd_dft, valacc_dft, valstd_dft \
        = none_vs_DFT321(classes, numTestFolds, numTrainFolds, title, *parameters)
        if (testacc_none == 1.0 or testacc_dft == 1.0): 
            text_file.write("\\rowcolor{TableHighlight} \n")
        text_file.write(re.sub(r"&", " and ", title))
        text_file.write(" & ")
        # for none
        text_file.write("%f & " %(optc_none))
        text_file.write("%0.4f (+/- %0.2f) & " %(valacc_none, valstd_none))
        text_file.write("%0.4f (+/- %0.2f) & " %(testacc_none, teststd_none))
        # for dft321
        text_file.write("%f & " %(optc_dft))
        text_file.write("%0.4f (+/- %0.2f) & " %(valacc_dft, valstd_dft))
        text_file.write("%0.4f (+/- %0.2f) " %(testacc_dft, teststd_dft))
        text_file.write("\\\\ \n")
        text_file.write("\hline ")
    text_file.close()

# Test frequency range

def none_vs_UpperFreqRange(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
        
    numObservations = len(label)
    
    # Test this C_range
    C_range = np.logspace(-5, 2, 20)
    # Test these upper frequency ranges
    freq_range = np.linspace(10, 20000, 100)
    
    # NONE
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs = np.zeros((numTestFolds, len(C_range)))
    OptC_none = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold
    
    
    # Frequency range reduction
    TestAccuracies_freq = np.zeros((numTestFolds, 1)) 
    ValAccuracies_freq = np.zeros((numTestFolds, 1)) 
    ValScores_freq = np.zeros((numTestFolds, len(freq_range))) 
    TrainScores_freq = np.zeros((numTestFolds, len(freq_range))) 
    ValStderrs_freq = np.zeros((numTestFolds, len(freq_range))) 
    TrainStderrs_freq = np.zeros((numTestFolds, len(freq_range)))
    OptC = np.zeros((numTestFolds, len(freq_range)))
    OptUpperFreq = np.zeros((numTestFolds, 1))
    OptC_freq = np.zeros((numTestFolds, 1))
    AvgTrainTime_freq = np.zeros((numTestFolds, 1)) # average training time per fold
    
    # For each fold, find an optimal C parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    
    # NONE
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptC]
        OptC_none[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        TrainScores[k,:] = train_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        TrainStderrs[k,:] = train_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_none[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    # Index for which outer fold we're in
    k = 0
    
    # Test upper frequency range
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(freq_range), len(C_range)))
        train_scores = np.zeros((len(freq_range), len(C_range)))
        val_stderrs = np.zeros((len(freq_range), len(C_range)))
        train_stderrs = np.zeros((len(freq_range), len(C_range)))
        avg_train_times = np.zeros((len(freq_range), len(C_range)))
        
        for i in range(len(freq_range)): # iterating through upper frequencies
            
            freq = freq_range[i]
            mod_data_train = data_train[:,:freq]
            mod_data_test = data_test[:,:freq]
            
            for ii in range(len(C_range)): # iterating through C values
                
                C = C_range[ii]

                # Split data_train into training and validation set, 5 fold cross-validation
                kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
                val_scores2 = np.zeros((kf2.n_folds, 1))
                train_scores2 = np.zeros((kf2.n_folds, 1))
                j=0
                total_train_time = 0
                for train2, val2 in kf2:

                    start_train_time = time.time()

                    # Define training and validation set
                    data_train2, data_val2, label_train2, label_val2 = mod_data_train[train2,:], mod_data_train[val2,:], label_train[train2], label_train[val2]

                    # Standardize the data
                    scaler2 = preprocessing.StandardScaler().fit(data_train2)
                    data_train_transformed2 = scaler2.transform(data_train2)
                    data_val_transformed2 = scaler2.transform(data_val2)

                    # Train and calculate validation score
                    clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=C).fit(data_train_transformed2, label_train2)
                    total_train_time += time.time() - start_train_time
                    val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                    train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                    j = j+1

                # record the average validation score for the specified number of components
                val_scores[i, ii] =  val_scores2.mean()
                train_scores[i, ii] = train_scores2.mean()
                val_stderrs[i, ii] = val_scores2.std()/np.sqrt(kf2.n_folds)
                train_stderrs[i, ii] = train_scores2.std()/np.sqrt(kf2.n_folds)
                avg_train_times[i, ii] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best number of components
        opt_c_per_freq = np.zeros((len(freq_range), 1))
        val_per_freq = np.zeros((len(freq_range), 1))
        train_per_freq = np.zeros((len(freq_range), 1))
        val_per_freq_std = np.zeros((len(freq_range), 1))
        train_per_freq_std = np.zeros((len(freq_range), 1))
        for i in range(len(val_scores)):
            indexOpt = np.argmax(val_scores[i,:])
            opt_c_per_freq[i] = C_range[indexOpt]
            val_per_freq[i] = val_scores[i,indexOpt]
            train_per_freq[i] = train_scores[i, indexOpt]
            val_per_freq_std[i] = val_stderrs[i, indexOpt]
            train_per_freq_std[i] = train_stderrs[i, indexOpt]
            
        ##########
        
        
        indexOptNumFreq = np.argmax(val_per_freq)
        ValAccuracies_freq[k] = val_per_freq[indexOptNumFreq]
        OptUpperFreq[k] = int(freq_range[indexOptNumFreq])
        OptC_freq[k] = opt_c_per_freq[indexOptNumFreq]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores_freq[k,:] = val_per_freq[:, 0]
        TrainScores_freq[k,:] = train_per_freq[:, 0] #########
        ValStderrs_freq[k,:] = val_per_freq_std[:, 0]
        TrainStderrs_freq[k,:] = train_per_freq_std[:, 0]
        AvgTrainTime_freq[k,:] = np.mean(np.mean(avg_train_times, 0), 0)
        OptC[k,:] = opt_c_per_freq[:,0]
        
        # Find the testing accuracy using the best C parameter
        
        mod_data_train = data_train[:,:OptUpperFreq[k]]
        mod_data_test = data_test[:,:OptUpperFreq[k]]
        
        scaler = preprocessing.StandardScaler().fit(mod_data_train)
        data_train_transformed = scaler.transform(mod_data_train)
        data_test_transformed = scaler.transform(mod_data_test)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_freq[k]).fit(data_train_transformed, label_train)
        TestAccuracies_freq[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print "Reporting with no dimensionality reduction (just DFT321)"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC_none.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    print "Reporting with Upper Frequency Range Cutoff"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime_freq, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal upper frequency cutoff is %d" %(int(OptUpperFreq.mean()))
    print "Optimal corresponding C is %f" %(OptC_freq.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies_freq.mean(), (TestAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies_freq.mean(), (ValAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96)
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylim([50,102])
    plt.axhline(y=ValAccuracies.mean()*100, xmin=0, xmax=1, hold=None, color='g', linestyle='-.', label='Best Val Acc w/o Dim Freq Range')
    te_a, caplines, errorlinecols = plt.errorbar(freq_range, np.mean(TrainScores_freq, 0)*100, yerr=np.mean(TrainStderrs_freq, 0)*100, fmt='bo-', label='Train Acc w/ Dim Freq Range')
    tr_a, caplines, errorlinecols = plt.errorbar(freq_range, np.mean(ValScores_freq, 0)*100, yerr=np.mean(ValStderrs_freq, 0)*100, fmt='ro-', label='Val Acc w/ Dim Freq Range')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.38),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('Upper Frequency Range')
    #plt.xscale('log')
    
    ax1.set_ylabel('Accuracy')
    
    ax2 = ax1.twinx()
    ax2.plot(freq_range, np.mean(OptC, 0), 'c')
    ax2.set_ylabel('Optimal C', color='c')
    
    plt.title('Effect of Frequency Range on Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images/FreqRange_vs_none_%s_%s.png' %(title, mode))
    
    return OptC_none.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96, OptUpperFreq.mean(), OptC_freq.mean(), \
    TestAccuracies_freq.mean(), (TestAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies_freq.mean(), (ValStderrs_freq.mean())*1.96

# Test frequency range

def none_vs_LowerFreqRange(classes, numTestFolds, numTrainFolds, upperFreq, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    objData = objData[:,:upperFreq]
        
    numObservations = len(label)
    
    # Test this C_range
    C_range = np.logspace(-5, 2, 20)
    # Test these lower frequency ranges
    freq_range = np.linspace(1, upperFreq-1, 50)
    
    # NONE
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs = np.zeros((numTestFolds, len(C_range)))
    OptC_none = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold
    
    
    # Frequency range reduction
    TestAccuracies_freq = np.zeros((numTestFolds, 1)) 
    ValAccuracies_freq = np.zeros((numTestFolds, 1)) 
    ValScores_freq = np.zeros((numTestFolds, len(freq_range))) 
    TrainScores_freq = np.zeros((numTestFolds, len(freq_range))) 
    ValStderrs_freq = np.zeros((numTestFolds, len(freq_range))) 
    TrainStderrs_freq = np.zeros((numTestFolds, len(freq_range)))
    OptC = np.zeros((numTestFolds, len(freq_range)))
    OptLowerFreq = np.zeros((numTestFolds, 1))
    OptC_freq = np.zeros((numTestFolds, 1))
    AvgTrainTime_freq = np.zeros((numTestFolds, 1)) # average training time per fold
    
    # For each fold, find an optimal C parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    
    # NONE
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptC]
        OptC_none[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        TrainScores[k,:] = train_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        TrainStderrs[k,:] = train_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_none[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    # Index for which outer fold we're in
    k = 0
    
    # Test lower frequency range
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(freq_range), len(C_range)))
        train_scores = np.zeros((len(freq_range), len(C_range)))
        val_stderrs = np.zeros((len(freq_range), len(C_range)))
        train_stderrs = np.zeros((len(freq_range), len(C_range)))
        avg_train_times = np.zeros((len(freq_range), len(C_range)))
        
        for i in range(len(freq_range)): # iterating through upper frequencies
            
            freq = freq_range[i]
            mod_data_train = data_train[:,freq:]
            mod_data_test = data_test[:,freq:]
            
            for ii in range(len(C_range)): # iterating through C values
                
                C = C_range[ii]

                # Split data_train into training and validation set, 5 fold cross-validation
                kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
                val_scores2 = np.zeros((kf2.n_folds, 1))
                train_scores2 = np.zeros((kf2.n_folds, 1))
                j=0
                total_train_time = 0
                for train2, val2 in kf2:

                    start_train_time = time.time()

                    # Define training and validation set
                    data_train2, data_val2, label_train2, label_val2 = mod_data_train[train2,:], mod_data_train[val2,:], label_train[train2], label_train[val2]

                    # Standardize the data
                    scaler2 = preprocessing.StandardScaler().fit(data_train2)
                    data_train_transformed2 = scaler2.transform(data_train2)
                    data_val_transformed2 = scaler2.transform(data_val2)

                    # Train and calculate validation score
                    clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=C).fit(data_train_transformed2, label_train2)
                    total_train_time += time.time() - start_train_time
                    val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                    train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                    j = j+1

                # record the average validation score for the specified number of components
                val_scores[i, ii] =  val_scores2.mean()
                train_scores[i, ii] = train_scores2.mean()
                val_stderrs[i, ii] = val_scores2.std()/np.sqrt(kf2.n_folds)
                train_stderrs[i, ii] = train_scores2.std()/np.sqrt(kf2.n_folds)
                avg_train_times[i, ii] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best number of components
        opt_c_per_freq = np.zeros((len(freq_range), 1))
        val_per_freq = np.zeros((len(freq_range), 1))
        train_per_freq = np.zeros((len(freq_range), 1))
        val_per_freq_std = np.zeros((len(freq_range), 1))
        train_per_freq_std = np.zeros((len(freq_range), 1))
        for i in range(len(val_scores)):
            indexOpt = np.argmax(val_scores[i,:])
            opt_c_per_freq[i] = C_range[indexOpt]
            val_per_freq[i] = val_scores[i,indexOpt]
            train_per_freq[i] = train_scores[i, indexOpt]
            val_per_freq_std[i] = val_stderrs[i, indexOpt]
            train_per_freq_std[i] = train_stderrs[i, indexOpt]
            
        ##########
        
        
        indexOptNumFreq = np.argmax(val_per_freq)
        ValAccuracies_freq[k] = val_per_freq[indexOptNumFreq]
        OptLowerFreq[k] = int(freq_range[indexOptNumFreq])
        OptC_freq[k] = opt_c_per_freq[indexOptNumFreq]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores_freq[k,:] = val_per_freq[:, 0]
        TrainScores_freq[k,:] = train_per_freq[:, 0] #########
        ValStderrs_freq[k,:] = val_per_freq_std[:, 0]
        TrainStderrs_freq[k,:] = train_per_freq_std[:, 0]
        AvgTrainTime_freq[k,:] = np.mean(np.mean(avg_train_times, 0), 0)
        OptC[k,:] = opt_c_per_freq[:,0]
        
        # Find the testing accuracy using the best C parameter
        
        mod_data_train = data_train[:,OptLowerFreq[k]:]
        mod_data_test = data_test[:,OptLowerFreq[k]:]
        
        scaler = preprocessing.StandardScaler().fit(mod_data_train)
        data_train_transformed = scaler.transform(mod_data_train)
        data_test_transformed = scaler.transform(mod_data_test)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_freq[k]).fit(data_train_transformed, label_train)
        TestAccuracies_freq[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print "Reporting with no dimensionality reduction (just DFT321)"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC_none.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    print "Reporting with Upper Frequency Range Cutoff"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime_freq, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal lower frequency cutoff is %d" %(int(OptLowerFreq.mean()))
    print "Optimal corresponding C is %f" %(OptC_freq.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies_freq.mean(), (TestAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies_freq.mean(), (ValAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96)
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylim([50,102])
    plt.axhline(y=ValAccuracies.mean()*100, xmin=0, xmax=1, hold=None, color='g', linestyle='-.', label='Best Val Acc w/o Dim Freq Range')
    te_a, caplines, errorlinecols = plt.errorbar(freq_range, np.mean(TrainScores_freq, 0)*100, yerr=np.mean(TrainStderrs_freq, 0)*100, fmt='bo-', label='Train Acc w/ Dim Freq Range')
    tr_a, caplines, errorlinecols = plt.errorbar(freq_range, np.mean(ValScores_freq, 0)*100, yerr=np.mean(ValStderrs_freq, 0)*100, fmt='ro-', label='Val Acc w/ Dim Freq Range')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.38),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('Lower Frequency Range')
    #plt.xscale('log')
    
    ax1.set_ylabel('Accuracy')
    
    ax2 = ax1.twinx()
    ax2.plot(freq_range, np.mean(OptC, 0), 'c')
    ax2.set_ylabel('Optimal C', color='c')
    
    plt.title('Effect of Frequency Range on Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images/FreqRangeLower_vs_none_%s_%s.png' %(title, mode))
    
    return OptC_none.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96, OptLowerFreq.mean(), OptC_freq.mean(), \
    TestAccuracies_freq.mean(), (TestAccuracies_freq.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies_freq.mean(), (ValStderrs_freq.mean())*1.96

def testAudioInterfaceFreqResponse(experiment, numComb, numTestFolds, numTrainFolds, lowFreq, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file_accel = open("TableData/AudioInterface_%s_%d_accel.txt" % (experiment, lowFreq), 'a')
    text_file_audio = open("TableData/AudioInterface_%s_%d_audio.txt" % (experiment, lowFreq), 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        # accelerometer
        optc, testacc, teststd, valacc, valstd \
        = SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "accel", *parameters)
        if (testacc == 1.0): 
            text_file_accel.write("\\rowcolor{TableHighlight} \n")
        text_file_accel.write(re.sub(r"&", " and ", title))
        text_file_accel.write(" & ")
        text_file_accel.write("%f & " %(optc))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_accel.write("\\\\ \n")
        text_file_accel.write("\hline ")
        # audio
        optc, testacc, teststd, valacc, valstd \
        = SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "audio", *parameters)
        if (testacc == 1.0): 
            text_file_audio.write("\\rowcolor{TableHighlight} \n")
        text_file_audio.write(re.sub(r"&", " and ", title))
        text_file_audio.write(" & ")
        text_file_audio.write("%f & " %(optc))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_audio.write("\\\\ \n")
        text_file_audio.write("\hline ")
    text_file_accel.close()
    text_file_audio.close()

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal C parameter
# Then it tests the C parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)
# This compares results when using PCA versus not. 

def none_vs_PCA(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
        
    numObservations = len(label)
    
    # Test this C_range
    C_range = np.logspace(-5, 2, 20)
    # Test these number of components in PCA
    numComponents = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # NONE
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs = np.zeros((numTestFolds, len(C_range)))
    OptC_none = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold
    
    
    # PCA
    TestAccuracies_PCA = np.zeros((numTestFolds, 1)) 
    ValAccuracies_PCA = np.zeros((numTestFolds, 1)) 
    ValScores_PCA = np.zeros((numTestFolds, len(numComponents))) 
    TrainScores_PCA = np.zeros((numTestFolds, len(numComponents))) 
    ValStderrs_PCA = np.zeros((numTestFolds, len(numComponents))) 
    TrainStderrs_PCA = np.zeros((numTestFolds, len(numComponents)))
    OptC = np.zeros((numTestFolds, len(numComponents)))
    OptNumComp = np.zeros((numTestFolds, 1))
    OptC_PCA = np.zeros((numTestFolds, 1))
    AvgTrainTime_PCA = np.zeros((numTestFolds, 1)) # average training time per fold
    
    # For each fold, find an optimal C parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    
        # Index for which outer fold we're in
    k = 0
    
    # NONE
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(data_train_transformed2)
                    data_train_transformed2 = pca.transform(data_train_transformed2)
                    data_val_transformed2 = pca.transform(data_val_transformed2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptC]
        OptC_none[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        TrainScores[k,:] = train_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        TrainStderrs[k,:] = train_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_none[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    # Index for which outer fold we're in
    k = 0
    
    # PCA
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(numComponents ), len(C_range)))
        train_scores = np.zeros((len(numComponents ), len(C_range)))
        val_stderrs = np.zeros((len(numComponents ), len(C_range)))
        train_stderrs = np.zeros((len(numComponents ), len(C_range)))
        avg_train_times = np.zeros((len(numComponents ), len(C_range)))
        
        for i in range(len(val_scores)): # iterating through components
            
            for ii in range(len(C_range)): # iterating through C values
                
                c = numComponents[i]
                C = C_range[ii]

                # Split data_train into training and validation set, 5 fold cross-validation
                kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
                val_scores2 = np.zeros((kf2.n_folds, 1))
                train_scores2 = np.zeros((kf2.n_folds, 1))
                j=0
                total_train_time = 0
                for train2, val2 in kf2:

                    start_train_time = time.time()

                    # Define training and validation set
                    data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]

                    # Standardize the data
                    scaler2 = preprocessing.StandardScaler().fit(data_train2)
                    data_train_transformed2 = scaler2.transform(data_train2)
                    data_val_transformed2 = scaler2.transform(data_val2)

                    # Implement PCA
                    pca = PCA(n_components = c)
                    pca.fit(data_train_transformed2)
                    data_train_transformed2 = pca.transform(data_train_transformed2)
                    data_val_transformed2 = pca.transform(data_val_transformed2)

                    # Train and calculate validation score
                    clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=C).fit(data_train_transformed2, label_train2)
                    total_train_time += time.time() - start_train_time
                    val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                    train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                    j = j+1

                # record the average validation score for the specified number of components
                val_scores[i, ii] =  val_scores2.mean()
                train_scores[i, ii] = train_scores2.mean()
                val_stderrs[i, ii] = val_scores2.std()/np.sqrt(kf2.n_folds)
                train_stderrs[i, ii] = train_scores2.std()/np.sqrt(kf2.n_folds)
                avg_train_times[i, ii] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best number of components
        opt_c_per_comp = np.zeros((len(numComponents), 1))
        val_per_comp = np.zeros((len(numComponents), 1))
        train_per_comp = np.zeros((len(numComponents), 1))
        val_per_comp_std = np.zeros((len(numComponents), 1))
        train_per_comp_std = np.zeros((len(numComponents), 1))
        for i in range(len(val_scores)):
            indexOpt = np.argmax(val_scores[i,:])
            opt_c_per_comp[i] = C_range[indexOpt]
            val_per_comp[i] = val_scores[i,indexOpt]
            train_per_comp[i] = train_scores[i, indexOpt]
            val_per_comp_std[i] = val_stderrs[i, indexOpt]
            train_per_comp_std[i] = train_stderrs[i, indexOpt]
            
        
        
        
        indexOptNumComp = np.argmax(val_per_comp)
        ValAccuracies_PCA[k] = val_per_comp[indexOptNumComp]
        OptNumComp[k] = int(numComponents[indexOptNumComp])
        OptC_PCA[k] = opt_c_per_comp[indexOptNumComp]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores_PCA[k,:] = val_per_comp[:, 0]
        TrainScores_PCA[k,:] = train_per_comp[:, 0] #########
        ValStderrs_PCA[k,:] = val_per_comp_std[:, 0]
        TrainStderrs_PCA[k,:] = train_per_comp_std[:, 0]
        AvgTrainTime_PCA[k,:] = np.mean(np.mean(avg_train_times, 0), 0)
        OptC[k,:] = opt_c_per_comp[:,0]
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA
        pca = PCA(n_components = int(OptNumComp[k]))
        pca.fit(data_train_transformed)
        data_train_transformed = pca.transform(data_train_transformed)
        data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC_PCA[k]).fit(data_train_transformed, label_train)
        TestAccuracies_PCA[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print "Reporting with no dimensionality reduction (just DFT321)"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC_none.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    print "Reporting with PCA"
    print("Average Train Time: %f" %(np.mean(AvgTrainTime_PCA, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal number of components is %d" %(int(OptNumComp.mean()))
    print "Optimal corresponding C is %f" %(OptC_PCA.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies_PCA.mean(), (TestAccuracies_PCA.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies_PCA.mean(), (ValAccuracies_PCA.std()/np.sqrt(kf1.n_folds))*1.96)
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylim([50,102])
    plt.axhline(y=ValAccuracies.mean()*100, xmin=0, xmax=1, hold=None, color='g', linestyle='-.', label='Best Validation Accuracy without PCA')
    te_a, caplines, errorlinecols = plt.errorbar(numComponents, np.mean(TrainScores_PCA, 0)*100, yerr=np.mean(TrainStderrs_PCA, 0)*100, fmt='bo-', label='Training Accuracy with PCA')
    tr_a, caplines, errorlinecols = plt.errorbar(numComponents, np.mean(ValScores_PCA, 0)*100, yerr=np.mean(ValStderrs_PCA, 0)*100, fmt='ro-', label='Validation Accuracy with PCA')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.38),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('Number of Components in PCA')
    plt.xscale('log')
    
    ax1.set_ylabel('Accuracy')
    
    ax2 = ax1.twinx()
    ax2.plot(numComponents, np.mean(OptC, 0), 'c')
    ax2.set_ylabel('Optimal C', color='c')
    
    plt.title('Effect of PCA on Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images/PCA_vs_none_%s_%s.png' %(title, mode))
    
    return OptC_none.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96, OptNumComp.mean(), OptC_PCA.mean(), \
    TestAccuracies_PCA.mean(), (TestAccuracies_PCA.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies_PCA.mean(), (ValStderrs_PCA.mean())*1.96

def run_none_vs_PCA(experiment, numComb, numTestFolds, numTrainFolds, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file_accel = open("TableData/nonevsPCA_%s_accel.txt" % experiment, 'a')
    text_file_audio = open("TableData/nonevsPCA_%s_audio.txt" % experiment, 'a')
    text_file_both = open("TableData/nonevsPCA_%s_both.txt" % experiment, 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        # accelerometer
        optc_none, testacc_none, teststd_none, valacc_none, valstd_none, \
        optnumcomp, optc_pca, testacc_pca, teststd_pca, valacc_pca, valstd_pca \
        = none_vs_PCA(classes, numTestFolds, numTrainFolds, title, "accel", *parameters)
        if (testacc_none == 1.0 or testacc_pca == 1.0): 
            text_file_accel.write("\\rowcolor{TableHighlight} \n")
        text_file_accel.write(re.sub(r"&", " and ", title))
        text_file_accel.write(" & ")
        # for none
        text_file_accel.write("%f & " %(optc_none))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(valacc_none, valstd_none))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(testacc_none, teststd_none))
        # for pca
        text_file_accel.write("%d & " %(int(optnumcomp)))
        text_file_accel.write("%f & " %(optc_pca))
        text_file_accel.write("%0.3f (+/- %0.2f) & " %(valacc_pca, valstd_pca))
        text_file_accel.write("%0.3f (+/- %0.2f) " %(testacc_pca, teststd_pca))
        text_file_accel.write("\\\\ \n")
        text_file_accel.write("\hline ")
        # audio
        optc_none, testacc_none, teststd_none, valacc_none, valstd_none, \
        optnumcomp, optc_pca, testacc_pca, teststd_pca, valacc_pca, valstd_pca \
        = none_vs_PCA(classes, numTestFolds, numTrainFolds, title, "audio", *parameters)
        if (testacc_none == 1.0 or testacc_pca == 1.0): 
            text_file_audio.write("\\rowcolor{TableHighlight} \n")
        text_file_audio.write(re.sub(r"&", " and ", title))
        text_file_audio.write(" & ")
        # for none
        text_file_audio.write("%f & " %(optc_none))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(valacc_none, valstd_none))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(testacc_none, teststd_none))
        # for pca
        text_file_audio.write("%d & " %(int(optnumcomp)))
        text_file_audio.write("%f & " %(optc_pca))
        text_file_audio.write("%0.3f (+/- %0.2f) & " %(valacc_pca, valstd_pca))
        text_file_audio.write("%0.3f (+/- %0.2f) " %(testacc_pca, teststd_pca))
        text_file_audio.write("\\\\ \n")
        text_file_audio.write("\hline ")
        # both
        optc_none, testacc_none, teststd_none, valacc_none, valstd_none, \
        optnumcomp, optc_pca, testacc_pca, teststd_pca, valacc_pca, valstd_pca \
        = none_vs_PCA(classes, numTestFolds, numTrainFolds, title, "both", *parameters)
        if (testacc_none == 1.0 or testacc_pca == 1.0): 
            text_file_both.write("\\rowcolor{TableHighlight} \n")
        text_file_both.write(re.sub(r"&", " and ", title))
        text_file_both.write(" & ")
        # for none
        text_file_both.write("%f & " %(optc_none))
        text_file_both.write("%0.3f (+/- %0.2f) & " %(valacc_none, valstd_none))
        text_file_both.write("%0.3f (+/- %0.2f) & " %(testacc_none, teststd_none))
        # for pca
        text_file_both.write("%d & " %(int(optnumcomp)))
        text_file_both.write("%f & " %(optc_pca))
        text_file_both.write("%0.3f (+/- %0.2f) & " %(valacc_pca, valstd_pca))
        text_file_both.write("%0.3f (+/- %0.2f) " %(testacc_pca, teststd_pca))
        text_file_both.write("\\\\ \n")
        text_file_both.write("\hline ")
    text_file_accel.close()
    text_file_audio.close()
    text_file_both.close()

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal C parameter
# Then it tests the C parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)

def SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
    
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
    if len(parameters) == 1:
        numComponents = parameters[0]
    if len(parameters) == 2:
        loFreq = parameters[0]
        hiFreq = parameters[1]
        objData = objData[:, loFreq:hiFreq]
    
    # Test this C_range
    C_range = np.logspace(-5, 2, 20)
    
    # Split data into folds
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C
    ValScores = np.zeros((numTestFolds, len(C_range))) # validation accuracies per C
    TrainScores = np.zeros((numTestFolds, len(C_range))) # training accuracies per C
    ValStderrs = np.zeros((numTestFolds, len(C_range))) 
    TrainStderrs = np.zeros((numTestFolds, len(C_range)))
    OptC = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold

    
    # For each fold, find an optimal C parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), 1))
        train_scores = np.zeros((len(C_range), 1))
        val_stderrs = np.zeros((len(C_range), 1))
        train_stderrs = np.zeros((len(C_range), 1))
        avg_train_times = np.zeros((len(C_range), 1))
        for i in range(len(val_scores)):
            
            c = C_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            train_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Standardize the data
                scaler2 = preprocessing.StandardScaler().fit(data_train2)
                data_train_transformed2 = scaler2.transform(data_train2)
                data_val_transformed2 = scaler2.transform(data_val2)
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(data_train_transformed2)
                    data_train_transformed2 = pca.transform(data_train_transformed2)
                    data_val_transformed2 = pca.transform(data_val_transformed2)
                
                # Train and calculate validation score
                clf2 = svm.SVC(decision_function_shape='ovo', kernel='linear', C=c).fit(data_train_transformed2, label_train2)
                total_train_time += time.time() - start_train_time
                val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                j = j+1
                
            # record the average validation score for the specified C parameter
            val_scores[i] =  val_scores2.mean()
            train_scores[i] = train_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            train_stderrs[i] = train_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds

            
        # Record the validation accuracy for the best C parameter
        indexOptC = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptC]
        OptC[k] = C_range[indexOptC]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        TrainScores[k,:] = train_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        TrainStderrs[k,:] = train_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best C parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='linear', C=OptC[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    fig1 = plt.gcf()
    axes = plt.gca()
    axes.set_ylim([50,102])
    te_a, caplines, errorlinecols = plt.errorbar(C_range, np.mean(TrainScores, 0)*100, yerr=np.mean(TrainStderrs, 0)*100, fmt='bo-', label='Training Accuracy')
    tr_a, caplines, errorlinecols = plt.errorbar(C_range, np.mean(ValScores, 0)*100, yerr=np.mean(ValStderrs, 0)*100, fmt='ro-', label='Validation Accuracy')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.32),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images2/BestLinearParameters_%s_%s.png' %(title, mode))

    return OptC.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96

def run_SVMDFT321_Linear_reportValAndTestAccuracy(experiment, numComb, numTestFolds, numTrainFolds, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file_accel = open("TableData/Linear_%s_accel.txt" % experiment, 'a')
    text_file_audio = open("TableData/Linear_%s_audio.txt" % experiment, 'a')
    text_file_both = open("TableData/Linear_%s_both.txt" % experiment, 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        # accelerometer
        optc, testacc, teststd, valacc, valstd \
        = SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "accel", *parameters)
        if (testacc == 1.0): 
            text_file_accel.write("\\rowcolor{TableHighlight} \n")
        text_file_accel.write(re.sub(r"&", " and ", title))
        text_file_accel.write(" & ")
        text_file_accel.write("%f & " %(optc))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_accel.write("\\\\ \n")
        text_file_accel.write("\hline ")
        # audio
        optc, testacc, teststd, valacc, valstd \
        = SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "audio", *parameters)
        if (testacc == 1.0): 
            text_file_audio.write("\\rowcolor{TableHighlight} \n")
        text_file_audio.write(re.sub(r"&", " and ", title))
        text_file_audio.write(" & ")
        text_file_audio.write("%f & " %(optc))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_audio.write("\\\\ \n")
        text_file_audio.write("\hline ")
        # both
        optc, testacc, teststd, valacc, valstd \
        = SVMDFT321_Linear_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "both", *parameters)
        if (testacc == 1.0): 
            text_file_both.write("\\rowcolor{TableHighlight} \n")
        text_file_both.write(re.sub(r"&", " and ", title))
        text_file_both.write(" & ")
        text_file_both.write("%f & " %(optc))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_both.write("\\\\ \n")
        text_file_both.write("\hline ")
    text_file_accel.close()
    text_file_audio.close()
    text_file_both.close()

# Utility function to move the midpoint of a colormap to be around the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal C parameter and gamma parameter
# Then it tests the C and gama parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)

def SVMDFT321_RBF_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
    
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
    if len(parameters) == 1:
        numComponents = parameters[0]
    if len(parameters) == 2:
        loFreq = parameters[0]
        hiFreq = parameters[1]
        objData = objData[:, loFreq, hiFreq]
    
    # Test this C_range and gamma_range
    C_range = np.logspace(-5, 10, 16)
    gamma_range = np.logspace(-9, 3, 13)
    
    # Split data into folds
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C and gamma
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C and gamma
    ValScores = np.zeros((numTestFolds, len(C_range), len(gamma_range))) # validation accuracies per C and gamma
    TrainScores = np.zeros((numTestFolds, len(C_range), len(gamma_range))) # training accuracies per C and gamma
    OptC = np.zeros((numTestFolds, 1))
    OptG = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold


    
    # For each fold, find an optimal C  and gamma parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=2)
    # Index for which outer fold we're in
    k = 0
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), len(gamma_range)))
        train_scores = np.zeros((len(C_range), len(gamma_range)))
        avg_train_times = np.zeros((len(C_range), len(gamma_range)))
        
        for c in range(len(C_range)):
            for g in range(len(gamma_range)):
                
                C = C_range[c]
                Gamma = gamma_range[g]
            
                # Split data_train into training and validation set, 5 fold cross-validation
                kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
                val_scores2 = np.zeros((kf2.n_folds, 1))
                train_scores2 = np.zeros((kf2.n_folds, 1))
                j=0
                total_train_time = 0
                for train2, val2 in kf2:
                    
                    start_train_time = time.time()
                
                    # Define training and validation set
                    data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                    # Standardize the data
                    scaler2 = preprocessing.StandardScaler().fit(data_train2)
                    data_train_transformed2 = scaler2.transform(data_train2)
                    data_val_transformed2 = scaler2.transform(data_val2)
                
                    # Implement PCA if valid
                    if len(parameters) == 1:
                        pca = PCA(n_components = numComponents)
                        pca.fit(data_train_transformed2)
                        data_train_transformed2 = pca.transform(data_train_transformed2)
                        data_val_transformed2 = pca.transform(data_val_transformed2)
                
                    # Train and calculate validation score
                    clf2 = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=C, gamma=Gamma).fit(data_train_transformed2, label_train2)
                    total_train_time += time.time() - start_train_time
                    val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                    train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                    j = j+1
                
                # record the average validation score for the specified C  and gamma parameter
                val_scores[c, g] =  val_scores2.mean()
                train_scores[c, g] = train_scores2.mean()
                avg_train_times[c, g] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best C and gamma parameter
        val_flattenedScores = val_scores.flatten()
        indexOpt = np.argmax(val_flattenedScores)
        ValAccuracies[k] = val_flattenedScores[indexOpt]
        OptC[k] = C_range[indexOpt/len(gamma_range)]
        OptG[k] = gamma_range[indexOpt%len(gamma_range)]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:,:] = val_scores
        TrainScores[k,:,:] = train_scores
        AvgTrainTime[k,:] = np.mean(np.mean(avg_train_times, 0), 0)
        
        # Find the testing accuracy using the best C and gamma parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', C=OptC[k], gamma=OptG[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC.mean())
    print "Optimal $\\gamma$ is %f" %(OptG.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    # Draw heatmap of the validation accuracy as a function of gamma and C
    plt.figure(figsize=(8,6))
    fig1 = plt.gcf()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(np.mean(ValScores, 0), interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('$\\gamma$')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Accuracy: %s - %s'%(title, mode))
    plt.show()
    plt.draw()
    fig1.savefig('../../images2/BestRBFParameters_%s_%s.png' %(title, mode))

    return OptC.mean(), OptG.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96

def run_SVMDFT321_RBF_reportValAndTestAccuracy(experiment, numComb, numTestFolds, numTrainFolds, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file_accel = open("TableData/RBF_%s_accel.txt" % experiment, 'a')
    text_file_audio = open("TableData/RBF_%s_audio.txt" % experiment, 'a')
    text_file_both = open("TableData/RBF_%s_both.txt" % experiment, 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        # accelerometer
        optc, optg, testacc, teststd, valacc, valstd \
        = SVMDFT321_RBF_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "accel", *parameters)
        if (testacc == 1.0): 
            text_file_accel.write("\\rowcolor{TableHighlight} \n")
        text_file_accel.write(re.sub(r"&", " and ", title))
        text_file_accel.write(" & ")
        text_file_accel.write("%f & " %(optc))
        text_file_accel.write("%f & " %(optg))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_accel.write("\\\\ \n")
        text_file_accel.write("\hline ")
        # audio
        optc, optg, testacc, teststd, valacc, valstd \
        = SVMDFT321_RBF_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "audio", *parameters)
        if (testacc == 1.0): 
            text_file_audio.write("\\rowcolor{TableHighlight} \n")
        text_file_audio.write(re.sub(r"&", " and ", title))
        text_file_audio.write(" & ")
        text_file_audio.write("%f & " %(optc))
        text_file_audio.write("%f & " %(optg))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_audio.write("\\\\ \n")
        text_file_audio.write("\hline ")
        # both
        optc, optg, testacc, teststd, valacc, valstd \
        = SVMDFT321_RBF_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "both", *parameters)
        if (testacc == 1.0): 
            text_file_both.write("\\rowcolor{TableHighlight} \n")
        text_file_both.write(re.sub(r"&", " and ", title))
        text_file_both.write(" & ")
        text_file_both.write("%f & " %(optc))
        text_file_both.write("%f & " %(optg))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_both.write("\\\\ \n")
        text_file_both.write("\hline ")
    text_file_accel.close()
    text_file_audio.close()
    text_file_both.close()

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal C parameter and number of degrees parameter
# Then it tests the C and numdegrees parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)

def SVMDFT321_Poly_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
    if len(parameters) == 1:
        numComponents = parameters[0]
    if len(parameters) == 2:
        loFreq = parameters[0]
        hiFreq = parameters[1]
        objData = objData[:, loFreq, hiFreq]
    
    # Test this C_range and numDeg_range
    C_range = np.logspace(-5, 10, 16)
    numDeg_range = range(2, 11)
    
    # Split data into folds
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal C and num degrees
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal C and num degrees
    ValScores = np.zeros((numTestFolds, len(C_range), len(numDeg_range))) # validation accuracies per C and num degrees
    TrainScores = np.zeros((numTestFolds, len(C_range), len(numDeg_range))) # training accuracies per C and num degrees
    OptC = np.zeros((numTestFolds, 1))
    OptD = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold


    
    # For each fold, find an optimal C  and num degrees parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=2)
    # Index for which outer fold we're in
    k = 0
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(C_range), len(numDeg_range)))
        train_scores = np.zeros((len(C_range), len(numDeg_range)))
        avg_train_times = np.zeros((len(C_range), len(numDeg_range)))
        
        for c in range(len(C_range)):
            for d in range(len(numDeg_range)):
                
                C = C_range[c]
                D = numDeg_range[d]
            
                # Split data_train into training and validation set, 5 fold cross-validation
                kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
                val_scores2 = np.zeros((kf2.n_folds, 1))
                train_scores2 = np.zeros((kf2.n_folds, 1))
                j=0
                total_train_time = 0
                for train2, val2 in kf2:
                    
                    start_train_time = time.time()
                
                    # Define training and validation set
                    data_train2, data_val2, label_train2, label_val2 = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                    # Standardize the data
                    scaler2 = preprocessing.StandardScaler().fit(data_train2)
                    data_train_transformed2 = scaler2.transform(data_train2)
                    data_val_transformed2 = scaler2.transform(data_val2)
                
                    # Implement PCA if valid
                    if len(parameters) == 1:
                        pca = PCA(n_components = numComponents)
                        pca.fit(data_train_transformed2)
                        data_train_transformed2 = pca.transform(data_train_transformed2)
                        data_val_transformed2 = pca.transform(data_val_transformed2)
                
                    # Train and calculate validation score
                    clf2 = svm.SVC(decision_function_shape='ovo', kernel='poly', C=C, degree=D).fit(data_train_transformed2, label_train2)
                    total_train_time += time.time() - start_train_time
                    val_scores2[j] = clf2.score(data_val_transformed2, label_val2)
                    train_scores2[j] = clf2.score(data_train_transformed2, label_train2)
                    j = j+1
                
                # record the average validation score for the specified C  and numDeg parameter
                val_scores[c, d] =  val_scores2.mean()
                train_scores[c, d] = train_scores2.mean()
                avg_train_times[c, d] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best C and gamma parameter
        val_flattenedScores = val_scores.flatten()
        indexOpt = np.argmax(val_flattenedScores)
        ValAccuracies[k] = val_flattenedScores[indexOpt]
        OptC[k] = C_range[indexOpt/len(numDeg_range)]
        OptD[k] = numDeg_range[indexOpt%len(numDeg_range)]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:,:] = val_scores
        TrainScores[k,:,:] = train_scores
        AvgTrainTime[k,:] = np.mean(np.mean(avg_train_times, 0), 0)
        
        # Find the testing accuracy using the best C and gamma parameter
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train_transformed = scaler.transform(data_train)
        data_test_transformed = scaler.transform(data_test)
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(data_train_transformed)
            data_train_transformed = pca.transform(data_train_transformed)
            data_test_transformed = pca.transform(data_test_transformed)
        clf = svm.SVC(decision_function_shape='ovo', kernel='poly', C=OptC[k], degree=OptD[k]).fit(data_train_transformed, label_train)
        TestAccuracies[k] = clf.score(data_test_transformed, label_test)
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal C is %f" %(OptC.mean())
    print "Optimal Number of Degrees is %f" %(OptD.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    # Draw heatmap of the validation accuracy as a function of gamma and numDegrees
    plt.figure(figsize=(8,6))
    fig1 = plt.gcf()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(np.mean(ValScores, 0), interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('Number of Degrees')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(numDeg_range)), numDeg_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Accuracy: %s - %s'%(title, mode))
    plt.show()
    plt.draw()
    fig1.savefig('../../images2/BestPolyParameters_%s_%s.png' %(title, mode))

    return OptC.mean(), OptD.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValAccuracies.std()/np.sqrt(kf1.n_folds))*1.96

def run_SVMDFT321_Poly_reportValAndTestAccuracy(experiment, numComb, numTestFolds, numTrainFolds, *parameters):
    a = list(range(len(experiment)-1))
    b = list(itertools.combinations(a, numComb))
    text_file_accel = open("TableData/Poly_%s_accel.txt" % experiment, 'a')
    text_file_audio = open("TableData/Poly_%s_audio.txt" % experiment, 'a')
    text_file_both = open("TableData/Poly_%s_both.txt" % experiment, 'a')

    for i in range(len(b)):
        classes = [experiment[b[i][0]]]
        title = experiment[b[i][0]]
        for j in range(1, numComb):
            classes = np.concatenate((classes, [experiment[b[i][j]]]), axis=0)
            title += '&%s' %(experiment[b[i][j]])
        # accelerometer
        optc, optd, testacc, teststd, valacc, valstd \
        = SVMDFT321_Poly_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "accel", *parameters)
        if (testacc == 1.0): 
            text_file_accel.write("\\rowcolor{TableHighlight} \n")
        text_file_accel.write(re.sub(r"&", " and ", title))
        text_file_accel.write(" & ")
        text_file_accel.write("%f & " %(optc))
        text_file_accel.write("%f & " %(optd))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_accel.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_accel.write("\\\\ \n")
        text_file_accel.write("\hline ")
        # audio
        optc, optg, testacc, teststd, valacc, valstd \
        = SVMDFT321_Poly_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "audio", *parameters)
        if (testacc == 1.0): 
            text_file_audio.write("\\rowcolor{TableHighlight} \n")
        text_file_audio.write(re.sub(r"&", " and ", title))
        text_file_audio.write(" & ")
        text_file_audio.write("%f & " %(optc))
        text_file_audio.write("%f & " %(optd))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_audio.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_audio.write("\\\\ \n")
        text_file_audio.write("\hline ")
        # both
        optc, optg, testacc, teststd, valacc, valstd \
        = SVMDFT321_Poly_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, "both", *parameters)
        if (testacc == 1.0): 
            text_file_both.write("\\rowcolor{TableHighlight} \n")
        text_file_both.write(re.sub(r"&", " and ", title))
        text_file_both.write(" & ")
        text_file_both.write("%f & " %(optc))
        text_file_both.write("%f & " %(optd))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(valacc, valstd))
        text_file_both.write("%0.4f (+/- %0.2f) & " %(testacc, teststd))
        text_file_both.write("\\\\ \n")
        text_file_both.write("\hline ")
    text_file_accel.close()
    text_file_audio.close()
    text_file_both.close()

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal alpha parameter
# Then it tests the alpha parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)

def SRCDFT321_LASSO_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
        
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
    if len(parameters) == 1:
        numComponents = parameters[0]
    if len(parameters) == 2:
        loFreq = parameters[0]
        hiFreq = parameters[1]
        objData = objData[:, loFreq, hiFreq]
    
    # Test this a_range
    a_range = np.logspace(-9, 0, 100)
    
    # Split data into folds
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal alpha
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal alpha
    ValScores = np.zeros((numTestFolds, len(a_range))) # validation accuracies per alpha
    ValStderrs = np.zeros((numTestFolds, len(a_range))) 
    OptA = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold

    
    # For each fold, find an optimal alpha parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(a_range), 1))
        val_stderrs = np.zeros((len(a_range), 1))
        avg_train_times = np.zeros((len(a_range), 1))
        for i in range(len(val_scores)):
            
            Alpha = a_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                D, Y, label_D, label_Y = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(D)
                    D = pca.transform(D)
                    Y = pca.transform(Y)
                    
                # Independently normalize each sample
                D = normalize(D)
                Y = normalize(Y)
                
                # Transform matrices so they are in the form of [numFeatures, numSamples]
                D = D.T # D, dictionary
                Y = Y.T # Y, the left out observations
                
                # Create W, the weights
                W = np.zeros((len(D[0,:]), len(Y[0,:])))
                # Create Y_hat_err, the square of the L_2 norm of the prediction error
                Y_hat_err = np.zeros((len(classes), 1))
                
                # Solve for w*
                clf = linear_model.Lasso(alpha=Alpha, max_iter=1000)
                for y in range(len(Y[0,:])):
                    clf.fit(D, Y[:,y])
                    W[:,y] = clf.coef_
                total_train_time += time.time() - start_train_time
                    
                # Find the closest prediction and score
                numCorrect = 0
                for y in range(len(Y[0,:])):
                    for obj in range(len(classes)):
                        numObsvPerClass = len(D[0,:]) / len(classes)
                        Y_hat = D[:, obj*numObsvPerClass:(obj+1)*numObsvPerClass].dot(W[obj*numObsvPerClass:(obj+1)*numObsvPerClass, y])
                        Y_hat_err[obj] = sum((Y[:,y] - Y_hat)**2)
                    numCorrect = numCorrect + (np.argmin(Y_hat_err) == label_Y[y])
                predAccuracy = (numCorrect + 0.0) / (len(Y[0,:]))
                
                val_scores2[j] = predAccuracy
                j = j+1
                
            # record the average validation score for the specified alpha parameter
            val_scores[i] =  val_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best alpha parameter
        indexOptA = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptA]
        OptA[k] = a_range[indexOptA]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best alpha parameter
        # data_train is now the dictionary and data_test is now Y
        
        D, Y, label_D, label_Y = data_train, data_test, label_train, label_test
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(D)
            D = pca.transform(D)
            Y = pca.transform(Y)
                    
        # Independently normalize each sample
        D = normalize(D)
        Y = normalize(Y)
                
        # Transform matrices so they are in the form of [numFeatures, numSamples]
        D = D.T # D, dictionary
        Y = Y.T # Y, the left out observations
                
        # Create W, the weights
        W = np.zeros((len(D[0,:]), len(Y[0,:])))
        # Create Y_hat_err, the square of the L_2 norm of the prediction error
        Y_hat_err = np.zeros((len(classes), 1))
                
        # Solve for w* using the optimal Alpha value found from the train/validation set
        clf = linear_model.Lasso(alpha=OptA[k], max_iter=1000)
        for y in range(len(Y[0,:])):
            clf.fit(D, Y[:,y])
            W[:,y] = clf.coef_
        
        # Find the closest prediction and score
        numCorrect = 0
        for y in range(len(Y[0,:])):
            for obj in range(len(classes)):
                numObsvPerClass = len(D[0,:]) / len(classes)
                Y_hat = D[:, obj*numObsvPerClass:(obj+1)*numObsvPerClass].dot(W[obj*numObsvPerClass:(obj+1)*numObsvPerClass, y])
                Y_hat_err[obj] = sum((Y[:,y] - Y_hat)**2)
            numCorrect = numCorrect + (np.argmin(Y_hat_err) == label_Y[y])
        
        TestAccuracies[k] = (numCorrect + 0.0) / (len(Y[0,:]))
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal $\\lambda$ is %f" %(OptA.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    fig1 = plt.gcf()
    axes = plt.gca()
    axes.set_ylim([0,102])
    tr_a, caplines, errorlinecols = plt.errorbar(a_range, np.mean(ValScores, 0)*100, yerr=np.mean(ValStderrs, 0)*100, fmt='ro-', label='Validation Accuracy')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.32),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title('Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images2/BestLASSOParameters_%s_%s.png' %(title, mode))
    
    return OptA.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96

# This function performs 2-fold cross validation to separate a testing and training+validation set
# Then it performs 5-fold cross validation to separate the training set into a training and validation set to find 
# an optimal alpha parameter
# Then it tests the alpha parameter on the testing set and reports an averaged testing accuracy (on each of the 2 folds)

def SRCDFT321_OMP_reportValAndTestAccuracy(classes, numTestFolds, numTrainFolds, title, mode, *parameters):
    
    start_time = time.time()

    objData_accel, objData_audio, label = prepDFT321(classes)
    objData = objData_accel
    if (mode == "audio"):
        objData = objData_audio
    if (mode == "both"):
        objData = np.concatenate((objData_accel, objData_audio), axis=1)
    numObservations = len(label)
    numComponents = 0
    
    # Reduce the dimensions of the data based on parameters
    if len(parameters) == 1:
        numComponents = parameters[0]
    if len(parameters) == 2:
        loFreq = parameters[0]
        hiFreq = parameters[1]
        objData = objData[:, loFreq, hiFreq]
    
    # Test this number of coefficients
    numObsvPerClass = int(len(objData[:,0]) / len(classes) * (1.0 - 1.0/numTestFolds) * (1.0 - 1.0/numTrainFolds))
    n_range = np.array(list(range(numObsvPerClass)))
    n_range = n_range + 1
    
    # Split data into folds
    TestAccuracies = np.zeros((numTestFolds, 1)) # testing accuracy corresponding to optimal alpha
    ValAccuracies = np.zeros((numTestFolds, 1)) # best validation accuracy corresponding to optimal alpha
    ValScores = np.zeros((numTestFolds, len(n_range))) # validation accuracies per alpha
    ValStderrs = np.zeros((numTestFolds, len(n_range))) 
    OptN = np.zeros((numTestFolds, 1))
    AvgTrainTime = np.zeros((numTestFolds, 1)) # average training time per fold

    
    # For each fold, find an optimal alpha parameter and test it on the other fold
    kf1 = StratifiedKFold(label, n_folds=numTestFolds)
    # Index for which outer fold we're in
    k = 0
    for train, test in kf1:
        
        # Define the training and testing set
        data_train, data_test, label_train, label_test = objData[train,:], objData[test,:], label[train], label[test]
        
        # Find an optimal parameter using data_train split into training and validation set
        val_scores = np.zeros((len(n_range), 1))
        val_stderrs = np.zeros((len(n_range), 1))
        avg_train_times = np.zeros((len(n_range), 1))
        for i in range(len(val_scores)):
            
            numCoefficients = n_range[i]
            
            # Split data_train into training and validation set, 5 fold cross-validation
            kf2 = StratifiedKFold(label_train, n_folds=numTrainFolds)
            val_scores2 = np.zeros((kf2.n_folds, 1))
            j=0
            total_train_time = 0
            for train2, val2 in kf2:
                
                start_train_time = time.time()
                
                # Define training and validation set
                D, Y, label_D, label_Y = data_train[train2,:], data_train[val2,:], label_train[train2], label_train[val2]
                
                # Implement PCA if valid
                if len(parameters) == 1:
                    pca = PCA(n_components = numComponents)
                    pca.fit(D)
                    D = pca.transform(D)
                    Y = pca.transform(Y)
                    
                # Independently normalize each sample
                D = normalize(D)
                Y = normalize(Y)
                
                # Transform matrices so they are in the form of [numFeatures, numSamples]
                D = D.T # D, dictionary
                Y = Y.T # Y, the left out observations
                
                # Create W, the weights
                W = np.zeros((len(D[0,:]), len(Y[0,:])))
                # Create Y_hat_err, the square of the L_2 norm of the prediction error
                Y_hat_err = np.zeros((len(classes), 1))
                
                # Solve for w*
                clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=int(numCoefficients))
                for y in range(len(Y[0,:])):
                    clf.fit(D, Y[:,y])
                    W[:,y] = clf.coef_
                total_train_time += time.time() - start_train_time
                    
                # Find the closest prediction and score
                numCorrect = 0
                for y in range(len(Y[0,:])):
                    for obj in range(len(classes)):
                        numObsvPerClass = len(D[0,:]) / len(classes)
                        Y_hat = D[:, obj*numObsvPerClass:(obj+1)*numObsvPerClass].dot(W[obj*numObsvPerClass:(obj+1)*numObsvPerClass, y])
                        Y_hat_err[obj] = sum((Y[:,y] - Y_hat)**2)
                    numCorrect = numCorrect + (np.argmin(Y_hat_err) == label_Y[y])
                predAccuracy = (numCorrect + 0.0) / (len(Y[0,:]))
                
                val_scores2[j] = predAccuracy
                j = j+1
                
            # record the average validation score for the specified alpha parameter
            val_scores[i] =  val_scores2.mean()
            val_stderrs[i] = val_scores2.std()/np.sqrt(kf2.n_folds)
            avg_train_times[i] = total_train_time/kf2.n_folds
            
        # Record the validation accuracy for the best alpha parameter
        indexOptN = np.argmax(val_scores)
        ValAccuracies[k] = val_scores[indexOptN]
        OptN[k] = n_range[indexOptN]
        
        # Save the validation and training accuracies to compare for generalization
        ValScores[k,:] = val_scores[:, 0]
        ValStderrs[k,:] = val_stderrs[:, 0]
        AvgTrainTime[k,:] = np.mean(avg_train_times, 0)
        
        # Find the testing accuracy using the best alpha parameter
        # data_train is now the dictionary and data_test is now Y
        
        D, Y, label_D, label_Y = data_train, data_test, label_train, label_test
        # Implement PCA if valid
        if len(parameters) == 1:
            pca = PCA(n_components = numComponents)
            pca.fit(D)
            D = pca.transform(D)
            Y = pca.transform(Y)
                    
        # Independently normalize each sample
        D = normalize(D)
        Y = normalize(Y)
                
        # Transform matrices so they are in the form of [numFeatures, numSamples]
        D = D.T # D, dictionary
        Y = Y.T # Y, the left out observations
                
        # Create W, the weights
        W = np.zeros((len(D[0,:]), len(Y[0,:])))
        # Create Y_hat_err, the square of the L_2 norm of the prediction error
        Y_hat_err = np.zeros((len(classes), 1))
                
        # Solve for w* using the optimal Alpha value found from the train/validation set
        clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=int(OptN[k]))
        for y in range(len(Y[0,:])):
            clf.fit(D, Y[:,y])
            W[:,y] = clf.coef_
        
        # Find the closest prediction and score
        numCorrect = 0
        for y in range(len(Y[0,:])):
            for obj in range(len(classes)):
                numObsvPerClass = len(D[0,:]) / len(classes)
                Y_hat = D[:, obj*numObsvPerClass:(obj+1)*numObsvPerClass].dot(W[obj*numObsvPerClass:(obj+1)*numObsvPerClass, y])
                Y_hat_err[obj] = sum((Y[:,y] - Y_hat)**2)
            numCorrect = numCorrect + (np.argmin(Y_hat_err) == label_Y[y])
        
        TestAccuracies[k] = (numCorrect + 0.0) / (len(Y[0,:]))
        k = k+1
        
    elapsed_time = time.time() - start_time
        
    # OUTPUT
    print("Elapsed Time: %f" %(elapsed_time))
    print("Average Train Time: %f" %(np.mean(AvgTrainTime, 0)))
    print "Averaged across %d folds:" %(kf1.n_folds)
    print "Optimal number of coefficients is %f" %(OptN.mean())
    print "Testing accuracy is %0.3f (+/- %0.2f)" %(TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96)
    print "Validation accuracy is %0.3f (+/- %0.2f)" %(ValAccuracies.mean(), (ValStderrs.mean())*1.96)
    fig1 = plt.gcf()
    axes = plt.gca()
    axes.set_ylim([50,102])
    tr_a, caplines, errorlinecols = plt.errorbar(n_range, np.mean(ValScores, 0)*100, yerr=np.mean(ValStderrs, 0)*100, fmt='ro-', label='Validation Accuracy')
    my_handler = HandlerLine2D(numpoints=2)
    plt.legend(handler_map={Line2D:my_handler}, bbox_to_anchor=(0.9, 0.32),
          bbox_transform=plt.gcf().transFigure, fancybox=True, framealpha=0.5)
    plt.xlabel('Number of coefficients')
    plt.ylabel('Accuracy')
    plt.title('Accuracy: %s - %s'%(title, mode))
    plt.show()
    fig1.savefig('../../images2/BestOMPParameters_%s_%s.png' %(title, mode))

    return OptN.mean(), TestAccuracies.mean(), (TestAccuracies.std()/np.sqrt(kf1.n_folds))*1.96, \
    ValAccuracies.mean(), (ValStderrs.mean())*1.96

### HERE ARE THE TESTS
none_vs_DFT321(glasses, 2, 5, 'glasses')
none_vs_DFT321(plastics, 2, 5, 'plastics')
none_vs_DFT321(papers, 2, 5, 'papers')
none_vs_DFT321(boxes, 2, 5, 'boxes')
none_vs_DFT321(nothings, 2, 5, 'nothings')
