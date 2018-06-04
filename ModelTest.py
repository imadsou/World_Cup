# Check the versions of libraries

# Python version
import sys

print('Python: {}'.format(sys.version))
# scipy
import scipy

print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np

print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd

print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn

print('sklearn: {}'.format(sklearn.__version__))

# Load libraries

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
#from sklearn import tree
#from sklearn import svm
from sklearn.cluster import KMeans
#from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import collections
dataset = pd.read_csv('Training_fifa.csv')

# shape
print(dataset.shape)

# head
print(dataset.head(5))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('label').size())

#print(dataset.groupby('class').size())
#dataset.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
#plt.show()


# histograms
dataset.hist()
#plt.show()

scatter_matrix(dataset)
#plt.show()
# Split-out validation dataset
array = dataset.values
X = dataset.loc[:,['rank_dif','rating_dif','goal_dif']]
#Y = dataset.loc[:,'label']
#Y2 = dataset.loc[:,['home_score','away_score']]
Y = np.vstack((dataset.loc[:,'home_score'], dataset.loc[:,'away_score'],dataset.loc[:,'label'])).T
#print("asd",Y)
#print('sss',Y2)
validation_size = 0.20
seed = 42
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

team_fer = pd.read_csv('team_features.csv')
print(team_fer.head())
def getval(team1,team2,year,df):
    team1_par=df[(df["nation"]==team1) & (df["year"]==year)].iloc[0]
    team2_par=df[(df["nation"]==team2) & (df["year"]==year)].iloc[0]
    #rank_dif=team1_par["Fifa Ranking"]-team2_par["Fifa Ranking"]
    goal_dif=team1_par["goal"]-team2_par["goal"]
    rank_dif=team1_par["Fifa Ranking"]-team2_par["Fifa Ranking"]
    rate_dif=team1_par["rating"]-team2_par["rating"]
	#goal_dif= team1_par["goal"]- team2_par["goal"]
    return rank_dif,rate_dif,goal_dif
ex1=getval('Argentina', 'Spain',2018,team_fer)
ex2=getval( 'Spain','Argentina',2018,team_fer)
#ex1=(-3, -4.9099999999999966)
#ex2=(3, 4.9099999999999966)
print(ex1)
# Spot Check Algorithms
models = []

#models.append(('1_LIN', linear_model.LinearRegression()))
#models.append(('2_LOR', LogisticRegression()))
#models.append(('3_TRI', tree.DecisionTreeClassifier(criterion='gini')))
models.append(('3_CART', DecisionTreeClassifier()))

#models.append(('4_SVM', SVC()))
#models.append(('svm', svm.SVC()))
#models.append(('5_GNB', GaussianNB()))
#models.append(('NB', GaussianNB()))

#models.append(('6_KNC', KNeighborsClassifier(5)))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('7_KMN', KMeans(n_clusters=9, random_state=42)))
#models.append(('8_RFC', RandomForestClassifier()))

#models.append(('10_GBC',  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)))

#models.append(('LDA', LinearDiscriminantAnalysis()))

# evaluate each model in turn
results = []
names = []
#print(collections.Counter(Y_validation))
for name, model in models:
	#kfold = model_selection.KFold(n_splits=10, random_state=seed)
	#cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	#results.append(cv_results)
	names.append(name)
	#model.fit(X_train, Y_train)
	#msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	#print(name)
	#print(model)
	#print(msg)

	#knn = KNeighborsClassifier()
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)





	#se = pd.Series(predictions)
	#print((se))

	#print(model.predict([list(ex1)])[0])
	#if model.score(X_train, Y_train) >0.6 :
	#print((model.predict([list(ex2)])[0]),(model.predict([list(ex1)])[0]),(name),model.score(X_train, Y_train))
	#print(model.score(X_train, Y_train))
	#print('Accuracy from sk-learn: {0}',format(model.score(X_train, Y_train)))
	#print(model.predict([list(ex1)])[0],name)
	#print(accuracy_score(Y_validation, predictions))
	#print(confusion_matrix(Y_validation, predictions))
	#print(classification_report(Y_validation, predictions))
	#print(collections.Counter(predictions))
	# save the model to disk
	print(model.predict([[4, 3.269999999999996, 1]])[0])
	print(model.predict([[-4, -3.269999999999996, -1]])[0])
	filename = name+'Final_fifa_Logmodel9.sav'
	pickle.dump(model, open(filename, 'wb'))


logreg = pickle.load(open('3_CARTFinal_fifa_Logmodel9.sav', 'rb'))
print(logreg.predict([[4, 3.269999999999996, 1]])[0])
print(logreg.predict([[-4, -3.269999999999996, -1]])[0])
df = pd.read_csv('Matches.csv')
#print(dataset2.head())

for index, row in df.iterrows():
	#print(row)
	exa=(getval(row['TeamA'], row['TeamB'], 2018, team_fer))
	A=((logreg.predict([list(exa)])[0])[0])
	B=((logreg.predict([list(exa)])[0])[1])
	C = ((logreg.predict([list(exa)])[0])[2])
	#if (logreg.predict([list(exa)])[0])>0 :
	df.at[index, 'ScoreA'] = A
	df.at[index, 'ScoreB'] = B
	df.at[index, 'Label'] = B
	#else:
	#df.at[index, 'ScoreA'] = 0
	#df.at[index, 'ScoreB'] = 1
	#print(df.head())

df.to_csv('out01.csv')
