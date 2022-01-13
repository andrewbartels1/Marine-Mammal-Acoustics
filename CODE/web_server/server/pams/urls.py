"""pams URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from ais.views import boat_locations
from .views import audio_clips, audio_file, buoy_locations, index, process_audio

urlpatterns = [
    path('', index),
    path('predict/', process_audio),
    path('ais/', boat_locations),
    path('buoys/', buoy_locations),
    path('audio_clips/', audio_clips),
    path('audio_file/', audio_file),
]
