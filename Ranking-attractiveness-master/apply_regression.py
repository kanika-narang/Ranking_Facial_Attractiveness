import numpy as np
import cv2
import pandas
from sklearn import linear_model

n_test=20
n=500
n_ratio=14


landmark = np.loadtxt('golden_ratio.txt', delimiter=' ')
landmark=np.reshape(landmark,(n,n_ratio))
print landmark.shape

for index in range(0,n,20):
    print np.arange(index,index+20)
    landmark_train=np.delete(landmark,np.arange(index,index+20),axis=0)
    print landmark_train.shape
    
'''
landmark_test = np.loadtxt('golden_ratio_test.txt', delimiter=' ')
landmark_test=np.reshape(landmark_test,(n_test,n_ratio))
rating = pandas.read_excel('Z:/IIITB/MP/Attractiveness/SCUT/Rating_Collection/Attractiveness_label.xlsx')
test_label=rating['Attractiveness label'][n_train:n_train+n_test+1]

#print rating['Attractiveness label']
train_label=rating['Attractiveness label'][0:n_train]
regr = linear_model.LinearRegression()
regr.fit(landmark_train, train_label)

ratings_predict = regr.predict(landmark_test)
print ratings_predict
print test_label
print len(ratings_predict)
print len(test_label)
#corr = np.corrcoef(np.rint(ratings_predict), np.rint(test_label))
print ratings_predict-test_label
'''
cv2.waitKey(0)
