#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selction import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def load_dataset(dataset_path):
	#To-Do: Implement this function
	return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	return dataset_df.data.shape[1],dataset_df.target[0].shape,dataset_df.target[1].shape

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	return train_test_split(dataset_df,dataset_df.target,test_size=testset_size)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	model=DecisionTreeClassifier()
	model.fit(x_train,y_train)
	return accuracy_score(y_test,model.predict(x_test)),precision_score(y_test,model.predict(x_test)),recall_score(y_test,model.predict(x_test))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	model=RandomForestClassifier()
	model.fit(x_train,y_train)
	return accuracy_score(model.predict(x_test),y_test),precision_score(y_test,model.predict(x_test)),recall_score(y_test,model.predict(x_test))

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	pipe=make_pipeline(
		StandardScaler(),
		SVC()
	)
	pipe.fit(x_train,y_train)
	return accuracy_score(y_test,pipe.predict(x_test)),precision_score(y_test,pipe.predict(x_test)),recall_score(y_test,pipe.predict(x_test))

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)