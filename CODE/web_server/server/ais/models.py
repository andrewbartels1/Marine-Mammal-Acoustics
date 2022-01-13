from django.db import models

# Create your models here.
class AISModel(models.Model):
    mmsi = models.IntegerField()
    base_date_time = models.DateTimeField()
    lat = models.FloatField()
    long = models.FloatField()
    vessel_type = models.IntegerField()
    boat_type = models.CharField(max_length=32)
