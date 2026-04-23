from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),  # <-- C'était ici l'erreur
    path('', include('core.urls')),
]