"""mpdevice URL Configuration

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
from django.contrib import admin
from django.urls import path
from rtsp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('extra_setup/', views.extra_setup),
    path('', views.main_page, name='main_page'),
    path('device/<str:device_name>/', views.main_page, name='main_page'),
    path('device_stream/<str:device_name>/<str:model>/<int:complexity>/<int:confidence>', views.device_url, name='device_stream'),
    path('close_device/', views.close_device, name='close_device'),
    path('add_new_rtsp/', views.add_new_rtsp, name='add_new_rtsp'),
    path('clear_setup/', views.clear_setup, name='clear_setup'),
    path('video/', views.no_livecam),
    path('video/<str:cam_name>/<str:model>/<int:complexity>/<int:confidence>', views.livecam_feed, name = 'livecam'),
]
