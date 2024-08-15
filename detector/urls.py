from django.urls import path
from .views import index

urlpatterns = [
    path('', index, name='index'),  # Maps the root URL of the app to the index view
]
