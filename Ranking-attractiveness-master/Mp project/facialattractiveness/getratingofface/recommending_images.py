import math
import operator
import sqlite3

def prepare_data(person):
	conn = sqlite3.connect('./db.sqlite3')
	rating_matrix=dict()
	for i in range(1,person+1):
		print "i is :", i
		val=int()
		rating_by_current_user=dict()
		val=i
		cursor = conn.execute("SELECT rating, imageid, user from UserRating where user=?",(val,))
		for row in cursor:
		   rating_by_current_user[row[1]]=row[0]
		rating_matrix[i]=rating_by_current_user
	conn.close()
	return rating_matrix

#returns common images between two persons
def common_images(person1,person2,ratingmatrix):
	common=[]
	n_images=ratingmatrix[person1]
	for image in n_images:
		if(image in ratingmatrix[person2]):
			common.append(image)
	return common

#returns euclidean distance between two persons calculated over common images
def euclidean_distance(person1,person2,common_image,ratingmatrix):
	d=0.0
	for image in common_image:
		x=ratingmatrix[person1][image]
		y=ratingmatrix[person2][image]
		d=d+math.pow(x-y,2)
	return 1/(1+d)

#pearson coefficient formula
def pearson_coefficient_formula(sigma_x,sigma_y,sigma_xsquare,sigma_ysquare,sigma_xy,n):
	numerator=sigma_xy-((sigma_x*sigma_y)/n)
	denominator=(math.sqrt(sigma_xsquare-((math.pow(sigma_x,2))/n)))*(math.sqrt(sigma_ysquare-((math.pow(sigma_y,2))/n)))
	if denominator!=0:
		coefficient=numerator/denominator
	else:
		coefficient=0
	return coefficient

#returns pearson coefficient of two persons calculated over common images, ranges between -1 to +1
def pearson_coefficient(person1,person2,common_image,ratingmatrix):
	sigma_xsquare=0
	sigma_ysquare=0
	sigma_x=0
	sigma_y=0
	sigma_xy=0
	n=0
	for image in common_image:
		x=ratingmatrix[person1][image]
		y=ratingmatrix[person2][image]
		sigma_x=sigma_x+x
		sigma_y=sigma_y+y
		sigma_xsquare=sigma_xsquare+math.pow(x,2)
		sigma_ysquare=sigma_ysquare+math.pow(y,2)
		sigma_xy=sigma_xy+(x*y)
		n=n+1
	if n!=0:
		coefficient=pearson_coefficient_formula(sigma_x,sigma_y,sigma_xsquare,sigma_ysquare,sigma_xy,n)
	else:
		coefficient=0
	return coefficient

# Calculate similarity between each pair of ratingmatrix using two type of metrics
# First is euclidean distance
# Second is Pearson Coefficient which gives linear correlation between two variables
def calculate_similarity(ratingmatrix):
	ratingmatrix_name=[]
	for i in ratingmatrix:
		ratingmatrix_name.append(i)
	for i in range(0,7):
		for j in range(i+1,7):
			person1=ratingmatrix_name[i]
			person2=ratingmatrix_name[j]
			common_image=common_images(person1,person2,ratingmatrix)
			#rel=euclidean_distance(person1,person2,common_image)
			#print person1,"and",person2,"has euclidean relation:",rel
			rel=pearson_coefficient(person1,person2,common_image,ratingmatrix)
			#print person1,"and",person2,"has pearson correlation:",rel


#returns list of images person has not rated
def images_not_rated(person,ratingmatrix):
	list_of_images_not_rated=set()
	for user,images in ratingmatrix.items():
		for image in images:
			if image not in ratingmatrix[person]:
				list_of_images_not_rated.add(image)
	return list_of_images_not_rated

def getRecommendations(person,ratingmatrix):
	#print ratingmatrix
	list_of_images_not_rated=images_not_rated(person,ratingmatrix)
	#print list_of_images_not_rated
	recommended_list_of_images=dict()
	similarity=dict()
	for user in ratingmatrix:
			if user != person:
				common_image=common_images(person,user,ratingmatrix)
				#print "common image with user ", user , "are" , common_image
				similarity[user]=pearson_coefficient(person,user,common_image,ratingmatrix)
				#print "similarity with this user",similarity[user]
	#print similarity
	for image in list_of_images_not_rated:
		total=0
		similarity_sum=0
		for user in ratingmatrix:
			if image in ratingmatrix[user]:
				#print user,image,person
				#print similarity[user],ratingmatrix[user][image],(similarity[user])*ratingmatrix[user][image]
				total=total+(similarity[user])*ratingmatrix[user][image]
				similarity_sum=similarity_sum+similarity[user]
		if similarity_sum!=0:
			recommendation_score=total/similarity_sum
		else:
			recommendation_score=0
		#print "recommendation_score of image", image, "is" , recommendation_score
		recommended_list_of_images[image]=recommendation_score
	#recommended_list_of_images.sort()
	#recommended_list_of_images.reverse()
	sorted_recommended_list_of_images = sorted(recommended_list_of_images.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_recommended_list_of_images

def getRecommendationForPerson(person):
	ratingmatrix=prepare_data(person)
	print ratingmatrix
	recommended_list_of_images=getRecommendations(person,ratingmatrix)
	to_return=list()
	if recommended_list_of_images:
		print "Following are the recommended list of images for" , person,":"
		for image in recommended_list_of_images:
			if image[1]>=4.0:
				print image[0], "is recommended with rating score", round(image[1],2)
				to_return.append(image[0])
	print to_return
	return to_return

#getRecommendationForPerson(person)