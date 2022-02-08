from django.contrib import admin
from .models import device, project, RTSP

admin.site.register(device)
admin.site.register(project)
admin.site.register(RTSP)
