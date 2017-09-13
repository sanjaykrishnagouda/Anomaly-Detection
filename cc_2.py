import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
import numpy as np
import time
import pandas as pd
import operator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV,train_test_split,KFold, cross_val_score
import sklearn,matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn.utils.testing import ignore_warnings, assert_raises
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


	X_train, X_test, y_train, y_test = train_test_split(input,output,test_size = 0.3, random_state = 0)

	X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,
		test_size = 0.3, random_state = 0)

	return(X_train,y_train,X_train_undersample,y_train_undersample)
@ignore_warnings
def printing_Kfold_scores(x_train_data,y_train_data): # LOGISTIC REGRESSION WITH 7 FOLD CV
	start = time.time()
	kf = KFold(n_splits = 7)
	c_param_range,t = [],0.0001
	#t=0.0001
	while t<=pow(10,1):
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

		recall_accs = []
		recall_accs_svm = []
		for iteration, (train,test) in enumerate(kf.split(x_train_data,y_train_data)):
			lr = LogisticRegression(C = c_param, penalty = 'l2')
			lr.fit(x_train_data.iloc[train],y_train_data.iloc[train].values.ravel())
			y_pred_undersample = lr.predict(x_train_data.iloc[test].values)
			# Calculate the recall score and append it to a list for recall scores representing the current c_parameter
			recall_acc = recall_score(y_train_data.iloc[test].values,y_pred_undersample)
			recall_accs.append(recall_acc)
	        

		recall_dict[c_param]=np.mean(recall_accs)
		results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)

		j += 1


	best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
	print("USING Logistic Regression::\nBest Mean: %f with inverse regularization strength %f"%(max(recall_dict.items(),key = operator.itemgetter(1))[1],
		max(recall_dict.items(),key = operator.itemgetter(1))[0]))


	lists2 = sorted(recall_dict.items())
	x2,y2 = zip(*lists2)
	plt.plot(x2,y2)
	plt.legend(['Logistic Regression'],loc='lower right')
	plt.draw()
	#return best_c,best_c_svm
	end = time.time()
	print("Logistic regressions took",end-start)
	return best_c


@ignore_warnings
def using_SVM(x_train_data,y_train_data,k):
	kernel = str(k)
	kf = KFold(n_splits = 5)
	c_param_range = []
	t=0.01
	while t<=2:
		c_param_range.append(t)
		t*=2


	results_table_svm = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
	results_table_svm['C_parameter'] = c_param_range
	# the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
	j = 0
	
	recall_dict_svm={}
	for c_param in c_param_range:
		
		#print('C parameter: ', c_param,'\t kernel',kernel)
		
		recall_accs_svm = []
		for iteration, (train,test) in enumerate(kf.split(x_train_data,y_train_data)):
			if kernel!='linear':
				clf = BaggingClassifier(SVC(C = c_param, kernel = kernel, verbose = False, max_iter = 1000),n_jobs=-1)
			if kernel == 'linear':
				clf = BaggingClassifier(LinearSVC(C = c_param, max_iter = 1000), n_jobs = 2)
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


@ignore_warnings
def diff_kerns(x,y):
	
	kernels = ['rbf','linear','sigmoid','poly']
	
	arr =[]
	for i in kernels:
		start = time.time()
		print("currently running ",i,"kernel")
		a=using_SVM(x,y,i)
		temp_dict,temp_best = a[0],a[1]
		arr.append(temp_dict[temp_best])
		end = time.time()
		print(i,"took %.2f seconds for completion."%(end-start))
	plt.scatter([1,2,3,4],arr)
	plt.xticks([1,2,3,4],kernels)
	plt.ylabel("Mean Recall of 7 iterations")
	
	plt.show()
	
def RandomForest(x,y):
	#first lets try GMM
	pass
def main(_):	
	a,b,c = load('creditcard.csv')
	temp = sampling_data(a,b,c)
	#printing_Kfold_scores(temp[0],temp[1]) # USING COMPLETE DATASET Logistic Regression
	diff_kerns(temp[0],temp[1]) # SVM with various kernels on complete dataset.
	#printing_Kfold_scores(temp[2],temp[3]) # USING UNDERSAMPLED DATASET Logistic Regression
	#diff_kerns(temp[2],temp[3]) # SVM with various kernels, run for 5 iterations
	#
if __name__ == "__main__":
	main(2)