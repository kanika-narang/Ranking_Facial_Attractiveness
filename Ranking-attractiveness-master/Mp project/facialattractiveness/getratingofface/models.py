# Create your models here.
from django.db import models

class ImageUpload(models.Model):
   picture = models.ImageField(upload_to = 'pictures')

   class Meta:
      db_table = "ImageUpload"


class UserRating(models.Model):
   rating=models.IntegerField()
   imageid=models.IntegerField()
   user=models.IntegerField()

   class Meta:
      db_table = "UserRating"   

class UserCount(models.Model):
	user=models.IntegerField()
	usernum=models.IntegerField()

	class Meta:
		db_table = "UserCount"

class Comment(models.Model):
	user=models.IntegerField()
	comment=models.TextField()

	class Meta:
		db_table="Comment"


class UserRatingForScud(models.Model):
	rating=models.IntegerField()
	imageid=models.IntegerField()
	user=models.IntegerField()

	class Meta:
		db_table ="UserRatingForScud"	
