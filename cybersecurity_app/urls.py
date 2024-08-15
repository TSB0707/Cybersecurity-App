from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detector/', include('detector.urls')),  # Include the URLs from the detector app
    path('', include('detector.urls')),  # Redirect root URL to the detector app
]
