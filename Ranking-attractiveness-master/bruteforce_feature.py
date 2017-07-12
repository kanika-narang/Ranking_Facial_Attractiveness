import numpy as np
import cv2
import itertools
import math
import pickle
from sklearn import decomposition
 

n=500
n_landmark=68

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def generateAllFeatures(allLandmarkCoordinates):
        #a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
        #a=list(range(68))
        a=[0,2,4,6,8,10,12,14,16,17,19,21,22,24,26,27,31,33,35,36,37,39,41,42,44,45,46,48,51,54,57,66]
        combinations = itertools.combinations(a, 4)
        i = 0
        pointIndices1 = [];
        pointIndices2 = [];
        pointIndices3 = [];
        pointIndices4 = [];
        for combination in combinations:
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[1])
            pointIndices3.append(combination[2])
            pointIndices4.append(combination[3])
            i = i+1
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[2])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[3])
            i = i+1
            pointIndices1.append(combination[0])
            pointIndices2.append(combination[3])
            pointIndices3.append(combination[1])
            pointIndices4.append(combination[2])
            i = i+1
        return generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates)

def facialRatio(points):
	x1 = points[0];
	y1 = points[1];
	x2 = points[2];
	y2 = points[3];
	x3 = points[4];
	y3 = points[5];
	x4 = points[6];
	y4 = points[7];
	dist1 = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	dist2 = math.sqrt((x3-x4)**2 + (y3-y4)**2)
	if isclose(dist2,0):ratio=0
	else:ratio = dist1/dist2
	return abs(ratio)

def generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates):
    size = allLandmarkCoordinates.shape
    allFeatures = np.zeros((size[0]/68, len(pointIndices1)))
   
    for i in range(0,len(landmark),68):
            lf=np.array(landmark[i:i+68])
            #print lf.shape
            ratios=[]
            print "face:"+str(i/68)            
            for j in range(0,len(pointIndices1)):
                    x1=lf[pointIndices1[j]][0]
                    y1=lf[pointIndices1[j]][1]                    
                    x2=lf[pointIndices2[j]][0]
                    y2=lf[pointIndices2[j]][1]
                    x3=lf[pointIndices3[j]][0]
                    y3=lf[pointIndices3[j]][1]
                    x4=lf[pointIndices4[j]][0]
                    y4=lf[pointIndices4[j]][1]
                    points = [x1, y1, x2, y2, x3, y3, x4, y4]
                    ratios.append(facialRatio(points))
            allFeatures[i/68, :] = np.asarray(ratios)
    print allFeatures.shape
    return allFeatures

landmark = np.loadtxt('landmarks.txt', delimiter=' ')
print landmark.size/68
bf_features= generateAllFeatures(landmark)
print "Before"
print bf_features.shape
#Applying PCA for dimensionality Reduction
pca = decomposition.PCA(n_components=25)
pca.fit(bf_features)
features_train = pca.transform(bf_features)
print features_train.shape
print features_train
filename = 'Asian_BF_PCA.sav'
pickle.dump(pca, open(filename, 'wb'))
    
#with open('Z:/IIITB/MP/Attractiveness/bruteforce_ratio_all.txt','a') as f_handle:
#        np.savetxt(f_handle,features_train,fmt='%.9f')
cv2.waitKey(0)
