from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^get_rating/$' , views.getRating, name='getRating'),
    url(r'^saved/', views.SaveImage, name = 'saved'),
    url(r'^userRating/', views.GetUserRating, name = 'userRating'),
    url(r'^recommend_faces/', views.GetRecommendedFaces, name = 'Recommend'),
    url(r'^listfaces/', views.ListFaces, name = 'listfaces'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG:
	urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)