import numpy as np
import pandas as pd
import operator
import tensorflow as tf
from sklearn import linear_model
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV,train_test_split,KFold, cross_val_score
import sklearn,matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
def load(str):
	data = pd.read_csv(str)
	# dropping two columns : 
	data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
	data = data.drop(['Time','Amount'], axis = 1)
	x = data.ix[:,data.columns != 'Class']
	y = data.ix[:,data.columns == 'Class']
	return data,x,y

def sampling_data(matrix, input, output):
	
	a = matrix[matrix.Class ==1]
	number_records_fraud = len(a)
	fraud_indices = np.array(a.index)

	# picking normal classes

	nonfraud_indices = matrix[matrix.Class == 0].index

	# selecting fraud number of normal samples
	random_normal_samples = np.random.choice(nonfraud_indices,
		number_records_fraud,replace = False)
	random_normal_samples = np.array(random_normal_samples)

	under_sample_indices = np.concatenate([fraud_indices,random_normal_samples])

	# collecting corresponding data

	under_sample_data = matrix.iloc[under_sample_indices,:]
	X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
	y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
	#print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
	#print("Resampled data:", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
	#print("Total number of transactions in resampled data: ", len(under_sample_data)) # so we now have equal number of 
																					  # fraud and normal examples!

	#splitting entire dataset into training and testing blocks

	X_train, X_test, y_train, y_test = train_test_split(input,output,test_size = 0.3, random_state = 0)
	#print('training block size: %i\ntesting block size: %i\ntotal: %i samples'%(len(X_train),len(X_test),len(X_train)+len(X_test)))

	#splitting undersampled dataset similarly
	X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,
		test_size = 0.3, random_state = 0)
	#print('For the undersampled data:\n training block size: %i\ntesting block size: %i\ntotal: %i samples'
		#%(len(X_train_undersample),len(X_test_undersample),len(X_train_undersample)+len(X_test_undersample)))

	return(X_train,y_train,X_train_undersample,y_train_undersample)

def printing_Kfold_scores(x_train_data,y_train_data):
   	
    kf = KFold(n_splits = 7)
    # Different C parameters, C = 1/lambda 
    #c_param_range = [0.0001,0.001,0.01,0.1,1,10,100]

    c_param_range = []
    
    t=0.0000001
    while t<=10:
    	c_param_range.append(t)
    	t*=10


    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
    results_table_svm = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table_svm['C_parameter'] = c_param_range
    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    recall_dict={}
    recall_dict_svm={}
    for c_param in c_param_range:
        #print('-------------------------------------------')
        #print('C parameter: ', c_param)
        #print('-------------------------------------------')
        #print('')

        recall_accs = []
        recall_accs_svm = []
        for iteration, (train,test) in enumerate(kf.split(x_train_data,y_train_data)):
        	
            lr = LogisticRegression(C = c_param, penalty = 'l2')
            lr.fit(x_train_data.iloc[train],y_train_data.iloc[train].values.ravel())
            #clf = SVC(C = c_param, kernel = 'rbf')
            #clf.fit(x_train_data.iloc[train],y_train_data.iloc[train].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[test].values)
            #y_pred_undersample_svm = clf.predict(x_train_data.iloc[test].values)


            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[test].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            
            #recall_acc_svm = recall_score(y_train_data.iloc[test].values,y_pred_undersample_svm)
            #recall_accs_svm.append(recall_acc_svm)
            #print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        
        recall_dict[c_param]=np.mean(recall_accs)
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        
        #recall_dict_svm[c_param]=np.mean(recall_accs_svm)
        #results_table_svm.ix[j,'Mean recall score'] = np.mean(recall_accs_svm)
        j += 1
        #print('')
        #print('Mean recall score using log reg', np.mean(recall_accs))
        #print('Mean recall score using SVM', np.mean(recall_accs_svm))
        #print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    print("USING Logistic Regression::\nBest Mean: %f with inverse regularization strength %.4f"%(max(recall_dict.items(),key = operator.itemgetter(1))[1],
    	max(recall_dict.items(),key = operator.itemgetter(1))[0]))

    #best_c_svm = results_table_svm.loc[results_table_svm['Mean recall score'].idxmax()]['C_parameter']
    #print("USING SVM :\nBest Mean: %f with inverse regularization strength %.4f"%(max(recall_dict_svm.items(),key = operator.itemgetter(1))[1],
    	#max(recall_dict_svm.items(),key = operator.itemgetter(1))[0]))
    # Finally, we can check which C parameter is the best amongst the chosen.
    #print('*********************************************************************************')
    #print('Best model to choose from cross validation is with C parameter = ', best_c)
    #print('*********************************************************************************')
    #print('Best Mean Recall Score is :%i\n'%(results_table[best_c]))
    #lists = sorted(recall_dict_svm.items())
    #x,y = zip(*lists)
    lists2 = sorted(recall_dict.items())
    x2,y2 = zip(*lists2)
    #plt.plot(x,y)
    #plt.ion()
    plt.plot(x2,y2)
    plt.legend(['Logistic Regression'],loc='lower right')
    plt.show()
    #return best_c,best_c_svm
    return best_c



def using_SVM(x_train_data,y_train_data,k):
    kernel = str(k)
    kf = KFold(n_splits = 5)
    c_param_range = []
    t=0.0001
    while t<=1000:
    	c_param_range.append(t)
    	t*=2


    results_table_svm = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table_svm['C_parameter'] = c_param_range
    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    recall_dict={}
    recall_dict_svm={}
    for c_param in c_param_range:
        
        #print('C parameter: ', c_param)
        
        recall_accs_svm = []
        for iteration, (train,test) in enumerate(kf.split(x_train_data,y_train_data)):
        	
            clf = SVC(C = c_param, kernel = kernel)
            clf.fit(x_train_data.iloc[train],y_train_data.iloc[train].values.ravel())

            y_pred_undersample_svm = clf.predict(x_train_data.iloc[test].values)
            
            recall_acc_svm = recall_score(y_train_data.iloc[test].values,y_pred_undersample_svm)
            recall_accs_svm.append(recall_acc_svm)
            
        
        recall_dict_svm[c_param]=np.mean(recall_accs_svm)
        results_table_svm.ix[j,'Mean recall score'] = np.mean(recall_accs_svm)
        j += 1
    

    best_c_svm = results_table_svm.loc[results_table_svm['Mean recall score'].idxmax()]['C_parameter']
    print("USING SVM :\nBest Mean: %f with inverse regularization strength %.4f using %s kernel"%(max(recall_dict_svm.items(),key = operator.itemgetter(1))[1],
    	max(recall_dict_svm.items(),key = operator.itemgetter(1))[0],kernel))
    return recall_dict_svm,best_c_svm



def diff_kerns(x,y):
	kernels = ['rbf','linear','sigmoid','poly']
	arr =[]
	for i in kernels:
		a=using_SVM(x,y,i)
		temp_dict,temp_best = a[0],a[1]
		arr.append(temp_dict[temp_best])
	#plt.hist(arr)
	#plt.xticks(arr,kernels)
	#plt.plot([1,2,3,4],arr)
	#plt.bar([1,2,3,4],arr, align = 'center')
	plt.scatter([1,2,3,4],arr)
	plt.xticks([1,2,3,4],kernels)
	plt.ylabel("Mean Recall of 7 iterations")
	plt.show()
	


a,b,c = load('creditcard.csv')
temp = sampling_data(a,b,c)
#printing_Kfold_scores(temp[0],temp[1]) # USING COMPLETE DATASET
printing_Kfold_scores(temp[2],temp[3]) # USING UNDERSAMPLED DATASET
#using_SVM(temp[2],temp[3])
diff_kerns(temp[2],temp[3])