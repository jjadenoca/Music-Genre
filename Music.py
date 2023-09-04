import numpy as np
import pandas as pd
import copy
#Splitting Data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
#Visualization
from matplotlib import pyplot as plt
import seaborn as sns
#Importing Data 
music_csv = pd.read_csv("music_genre.csv")
music = pd.DataFrame(music_csv)
music_df = music.copy()

#Previewing Data
print(music_df.head())
print(music_df.columns)
print(music_df['music_genre'].unique()) 

#Data Cleaning 
print(pd.isnull(music_df).values.sum()) #90 null values in the data frame
print(music_df.index[music_df.isnull().any(axis=1)]) #Finding null values in df
music_df = music_df.dropna(how='any',axis=0) 
print(pd.isnull(music_df).values.sum()) #no more null values in the df
music_df.drop(['instance_id', 'artist_name', 'track_name', 'obtained_date'], axis = 1, inplace = True)
for column in music_df:
    print(type(column))

#? also represents missing data in the, i want to remove entries with missing data
#print(music_df.index[music])
print(music_df.columns)
print(type(music_df['key'][0]))
music_df.rename(columns = {'mode': 'keymode'}, inplace=True)
print(music_df.columns)
#Finding a way to objectify key signatures based on their position in the circle of fifths
#Ranking them 1 to 12 C G D A E B F# C# G# D# A# F

circle_of_fifths = {'C':0, 'G':1, 'D':2, 'A':3, 'E':4, 'B':5, 'F#':6, 'C#':7, 'G#':8, 'D#':9, 'A#':10, 'F':11}
key_mode = {'Major': 0, 'Minor': 1}
music_df = music_df.assign(keymode=music_df.keymode.map(key_mode))
print(music_df['key'].unique())

music_df = music_df.assign(key=music_df.key.map(circle_of_fifths))
    # key = music_df['key'][num]
    # print(key)
    # print(type(key))
    # music_df['key'][num] = circle_of_fifths.index(key)
    # key = music_df['key'][num]
    # print(key)
    # print(type(key))

music_df = music_df[music_df.tempo != '?']
music_df = music_df[music_df.duration_ms != -1]
music_genre = music_df['music_genre']



music_df = music_df.drop('music_genre', axis=1)
music_df = music_df.astype(np.float64, copy = True, errors = 'raise')

print(np.sort(music_df['key'].unique()))
print(type(music_df['key']))
print(music_df.head())


print(music_df['tempo'])
# print('tempo:{}'.format(type(music_df['tempo'][0])))
# print('valence:{}'.format(type(music_df['valence'][0])))

tempValCorr = music_df['tempo'].corr(music_df['valence'])
print('Correlation between tempo (speed) and valence (how happy the song feels): {}'.format(tempValCorr))
X = music_df.copy()#Drop it from independent variables because this is what we aere trying to predict
Y = music_genre
print(music_df.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, test_size = 0.3, random_state= 3434)

mode_onehot_pipe = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most frequent')),
    ('one hot encoder', OneHotEncoder(handle_unknown='ignore'))])

transformer = ColumnTransformer([('one hot', OneHotEncoder(handle_unknown='ignore'), ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'keymode',
       'speechiness', 'tempo', 'valence']),], remainder='passthrough')


print("Model creation")
gradient = GradientBoostingClassifier(random_state=3434)
logreg = LogisticRegression(random_state = 3434, max_iter=1000)
linsvc = LinearSVC(random_state = 3434, max_iter=1000)
dct = DecisionTreeClassifier(random_state = 3434)
knn = KNeighborsClassifier()
grad_pipe = Pipeline([('transformer', transformer), ('gradient', gradient)])
logreg_pipe = Pipeline([('transformer', transformer), ('logreg', logreg)])
linsvc_pipe = Pipeline([('transformer', transformer), ('linsvc', linsvc)])
dct_pipe = Pipeline([('transformer', transformer), ('dct', dct)])
knn_pipe = Pipeline([('transformer', transformer), ('knn', knn)])
for model in [grad_pipe, logreg_pipe, linsvc_pipe, dct_pipe, knn_pipe]:
    print("training next model")
    model.fit(X_train, Y_train)
accuracy_scores = [accuracy_score(Y_test, grad_pipe.predict(X_test)),
                   accuracy_score(Y_test, logreg_pipe.predict(X_test)),
                   accuracy_score(Y_test, linsvc_pipe.predict(X_test)),
                   accuracy_score(Y_test, dct_pipe.predict(X_test)),
                   accuracy_score(Y_test, knn_pipe.predict(X_test))]

model_names = ["Gradient Boosting Classifier", "Logistic Regression" , "Linear SVC", "Decision Tree Classifier", "KNN Classifier"]

model_summary = pd.DataFrame({'Model Method': model_names, 'Accuracy': accuracy_scores})
print(model_summary)
plt.scatter(music_df['tempo'], music_df['energy'], s = 0.01)
plt.show()







