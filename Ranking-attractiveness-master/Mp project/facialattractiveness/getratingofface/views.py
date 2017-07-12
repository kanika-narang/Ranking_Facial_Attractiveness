from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from .forms import UploadFileForm
from .models import UserRating , UserCount ,UserRatingForScud,  Comment
import json,requests
from facialattractiveness import settings
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
from sklearn import decomposition
import dlib
import itertools
import math
import pickle
import warnings
import recommending_images
warnings.filterwarnings("ignore")


cascade_path = "D:/Software Setups/Python package/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
predictor_path= "D:/Software Setups/dlib-master/python_examples/shape_predictor_68_face_landmarks.dat"
GR_SVM_As=0
GR_SVR_As=0.0
GR_LR_As=0.0
GR_LR_PCA_As=0.0
GR_SVM_PCA_As=0

from getratingofface.forms import UploadFileForm
from getratingofface.models import ImageUpload

def SaveImage(request):
   saved = False
   context={}
   if request.method == "POST":
      #Get the posted form
      MyImageForm = UploadFileForm(request.POST, request.FILES)
      
      if MyImageForm.is_valid():
         image = ImageUpload()
         imagepic= MyImageForm.cleaned_data["picture"]
         image.picture =imagepic
         image.save()
         saved = True
         context["imageurl"]=imagepic
         getLandmark()
   else:
      MyImageForm = UploadFileForm()
   
   context["GR_LR_As"]=GR_LR_As
   context["GR_SVR_As"]=GR_SVR_As
   context["GR_SVM_As"]=GR_SVM_As
	# print "nice"
   #HTMLResponse = render_to_string('datatable.html',context)
			
   return render(request, 'datatable.html', context)

def GetRecommendedFaces(request):
	recommendedimages=""
	if request.method=="POST":
		t = UserCount.objects.get(user=1)
		t.usernum=t.usernum+1
		usernewid=t.usernum
		print("User Id is "+str(usernewid))
		t.save()
		print("saved the User Id")

		for i in range(1,151):
			rating=request.POST.get('rating'+str(i)+'')
			if(rating!=None):
				userratingmodel=UserRating()
				userratingmodel.rating=rating
				userratingmodel.imageid=i
				userratingmodel.user=usernewid
				userratingmodel.save()
		print("done..........")
		print("Ratings Saved")	
		recommendedimages=recommending_images.getRecommendationForPerson(usernewid)
		recommendedimages = json.dumps(recommendedimages)
		print recommendedimages
	return render(request, 'recommendedlist.html', {"recommendedimages" : recommendedimages})


def GetUserRating(request):
	if request.method=="POST":
		for i in range(1,200):
			rating=request.POST.get('rating'+str(i)+'')
			if(rating!=""):
				userratingmodel=UserRating()
				userratingmodel.rating=(request.POST.get('rating'+str(i)+''))
				userratingmodel.imageid=i
				userratingmodel.save()
		print("done..........")
	recommendedimages=recommending_images.getRecommendationForPerson(usernewid)
	recommendedimages = json.dumps(recommendedimages)
	print recommendedimages
	return render(request, 'recommendedlist.html', {"recommendedimages" : recommendedimages})		

	context["recarray"]=GetRecommendation()	
	return render(request, 'RecommendedFaces.html', locals())

# Create your views here.
def index(request):
	print("hello")
	return render(request, "index.html")

@csrf_exempt
def getRating(request):
	print("uploaded image")
	if request.method == "POST":
		file=request.POST.get('filelocation')
		print(file)
	print("done")	
	getLandmark(file)
	return HttpResponse("Done sending")

def getLandmark():
	print("here")
	image=ImageUpload.objects.order_by('-id')[0]
	image_url=image.picture.url
	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascade_path)
	# create the landmark predictor
	predictor = dlib.shape_predictor(predictor_path)
	landmark_text=np.array([])
	semipath=settings.MEDIA_ROOT
	image_path=str(semipath)+str(image_url)
	image_path=image_path.replace("/media","",1)
	image_path=image_path.replace("/","\\")
        #image_path = image_url
        print "Image................................"
        print image_path
        # Read the image
        image = cv2.imread(image_path)
       #cv2.imshow("img",image)
        
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        #print "Found {0} faces!".format(len(faces))
        #print faces
        x=faces[0,0]
        y=faces[0,1]
        w=faces[0,2]
        h=faces[0,3]
        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        #print dlib_rect
        #print "getting Landmarks"
        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # copying the image so we can see side-by-side
        image_copy = image.copy()
        #print landmarks
        for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(image_copy, pos, 3, color=(0, 0, 255),thickness=-1)
        np.set_printoptions(suppress=True)
        #cv2.imshow("Landmarks found", image_copy)
        #print "Sving..."
        #with open('Z:/IIITB/MP/Attractiveness/this_landmarks.txt','a') as f_handle:
        #        np.savetxt(f_handle,landmarks,fmt='%d')
        get_GRRatio(landmarks)
        #get_BFRatio(landmarks)

def get_GRRatio(landmark):
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
	        get_ModelLinearRegressionRating(golden_ratio)
	        get_ModelSVRRating(golden_ratio)
	        get_ModelSVMRating(golden_ratio)


#Brute force features

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
   
    for i in range(0,len(allLandmarkCoordinates),68):
            lf=np.array(allLandmarkCoordinates[i:i+68])
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


def get_BFRatio(landmark):
	bf_features= generateAllFeatures(landmark)
	#Applying PCA for dimensionality Reduction
	#pca = decomposition.PCA(n_components=25)
	#pca.fit(bf_features)
	print bf_features.size
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_BF_PCA.sav', 'rb'))
	features_train = loaded_model.transform(bf_features)
	#features_train = pca.transform(bf_features)
	get_ModelLinearRegressionPCARating(features_train)
	get_ModelSVMPCARating(features_train)


def get_ModelLinearRegressionRating(landmark):
	#Linear Regression Model
	global GR_LR_As
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_LinearRegression.sav', 'rb'))
	result = loaded_model.predict(landmark)
	GR_LR_As=result
	print result
	
def get_ModelSVRRating(landmark):
	global GR_SVR_As
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_SVR.sav', 'rb'))
	result = loaded_model.predict(landmark)
	GR_SVR_As=result
	print result

def get_ModelSVMRating(landmark):
	global GR_SVM_As
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_SVC.sav', 'rb'))
	result = loaded_model.predict(landmark)
	GR_SVM_As=result
	print result

def get_ModelLinearRegressionPCARating(landmark):
	#Linear Regression Model
	global GR_LR_PCA_As
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_LinearRegression.sav', 'rb'))
	result = loaded_model.predict(landmark)
	GR_LR_PCA_As=result
	print result
	

def get_ModelSVMPCARating(landmark):
	global GR_SVM_PCA_As
	loaded_model = pickle.load(open('Z:/IIITB/MP/Attractiveness/Asian_SVC.sav', 'rb'))
	result = loaded_model.predict(landmark)
	GR_SVM_PCA_As=result
	print result



def getDataToShow(request):
	print "here"
	HTMLResponse = ""
	datatoshow=""
	responseMessage = {}
	error = {}
	context = {}
	print "inside"
	try:
		print "good"
	except Exception, e:
		error["custom"] =  str(e)
	#responseMessage['htmlresponse'] =HTMLResponse 
	context["GR_LR_As"]=GR_LR_As
	context["GR_SVR_As"]=GR_SVR_As
	context["GR_SVM_As"]=GR_SVM_As
	context["GR_LR_PCA_As"]=GR_LR_PCA_As
	context["GR_SVM_PCA_As"]=GR_SVM_PCA_As
	# print "nice"
	HTMLResponse = render_to_string('datatable.html',context)
	responseMessage['datatoshow']=HTMLResponse
	responseMessage['error'] = error
	return HttpResponse(str(json.dumps(responseMessage)))    	
	return 0


def GetRecommendation():
	arr=[1,2,3,4,5]
	return arr


def ListFaces(request):
	return render(request, "recommendation.html")	