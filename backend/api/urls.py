from django.urls import path, include

urlpatterns = [
    path('hati/', include('api.hati.urls'))
]