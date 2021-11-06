from django.urls import path

# from .views import HatiMetaView, LookMapView, HatiFindView, UploadPhotoView
from .views import HatiMetaView, LookMapView, HatiFindView

urlpatterns = [
    path('meta/', HatiMetaView.as_view()),
    path('map/', LookMapView.as_view()),
    path('advertisement/', HatiFindView.as_view()),
    # path('upload/', UploadPhotoView.as_view()),
]
