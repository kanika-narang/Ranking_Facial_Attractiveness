import numpy as np
import cv2

landmark = np.loadtxt('landmarks_test.txt', delimiter=' ')
print len(landmark)/68


golden_ratio=np.array([])
for i in range(0,len(landmark),68):
        #lf contains landmarks of single face
        lf=np.array(landmark[i:i+68])
        #print len(landmark_face)
        ratio=np.array([])

        #Calculate 14 ratios
        #Mideye distance to interocular distance
        num=((lf[41,0]+lf[40,0])/2)-((lf[47,0]+lf[46,0])/2)
        den=(lf[42,0]-lf[39,0])
        ratio=np.append(ratio,abs(num/den))
        #Mideye distance to nose width
        num=((lf[41,0]+lf[40,0])/2)-((lf[47,0]+lf[46,0])/2)
        den=(lf[31,0]-lf[35,0])
        ratio=np.append(ratio,abs(num/den))
        #Mouth width to interocular distance
        num=lf[48,0]-lf[54,0]
        den=(lf[42,0]-lf[39,0])
        ratio=np.append(ratio,abs(num/den))
        #Lip chin distance to interocular distance
        num=lf[51,1]-lf[8,1]
        den=(lf[42,0]-lf[39,0])
        ratio=np.append(ratio,abs(num/den))
        #Lip chin distance to nose width
        num=lf[51,1]-lf[8,1]
        den=(lf[31,0]-lf[35,0])
        ratio=np.append(ratio,abs(num/den))
        #Interocular distance to eye fisher width
        num=lf[42,0]-lf[39,0]
        den=(lf[42,0]-lf[45,0])
        ratio=np.append(ratio,abs(num/den))
        #Interocular distance to lip height
        num=lf[42,0]-lf[39,0]
        den=(lf[51,1]-lf[57,1])
        ratio=np.append(ratio,abs(num/den))
        #Nose width to eye fisher width
        num=lf[31,0]-lf[35,0]
        den=(lf[42,0]-lf[45,0])
        ratio=np.append(ratio,abs(num/den))
        #Nose width to lip height
        num=lf[31,0]-lf[35,0]
        den=(lf[51,1]-lf[57,1])
        ratio=np.append(ratio,abs(num/den))
        #Eye fisher width to Nose mouth distance
        num=lf[42,0]-lf[45,0]
        den=(lf[33,1]-lf[66,1])
        ratio=np.append(ratio,abs(num/den))
        #Lip height to nose mouth distance
        num=lf[51,1]-lf[57,1]
        den=(lf[33,1]-lf[66,1])
        ratio=np.append(ratio,abs(num/den))
        #Nose chin distance to Lip chin distance
        num=lf[33,1]-lf[8,1]
        den=(lf[66,1]-lf[8,1])
        ratio=np.append(ratio,abs(num/den))
        #Nose width to nose mouth
        num=lf[31,0]-lf[35,0]
        den=(lf[33,1]-lf[66,1])
        ratio=np.append(ratio,abs(num/den))
        #Mouth width to nose width
        num=lf[48,0]-lf[54,0]
        den=(lf[31,0]-lf[35,0])
        ratio=np.append(ratio,abs(num/den))
        golden_ratio=np.append(golden_ratio,ratio)
        
with open('Z:/IIITB/MP/Attractiveness/golden_ratio_test.txt','a') as f_handle:
        np.savetxt(f_handle,golden_ratio,fmt='%.9f')
cv2.waitKey(0)
