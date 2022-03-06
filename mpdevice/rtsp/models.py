from django.db import models
import json
# Create your models here.

class project(models.Model):
    name = models.CharField(max_length=50, null=False)
    def __str__(self):
        return self.name

class RTSP(models.Model):
    name = models.CharField(max_length=50, null=False)
    location = models.CharField(max_length=100, null=False)
    url = models.TextField(null=False)
    def __str__(self):
        return self.url
    
class device(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50, null=False)
    model = models.CharField(max_length=20, null=False)
    complexity = models.DecimalField(max_digits=4, decimal_places=2, null=False, default=0)
    confidence = models.DecimalField(max_digits=4, decimal_places=2, null=False, default=0.5)
    project = models.ForeignKey(project, on_delete=models.CASCADE, null=True)
    rtsp = models.ForeignKey(RTSP, on_delete=models.PROTECT)
    uuid = models.CharField(max_length=36, null=True)
    
    def __str__(self):
        return self.name
    def set_name(self, x):
        self.class_name = json.dumps(x)
    def get_name(self):
        return json.loads(self.class_name)
