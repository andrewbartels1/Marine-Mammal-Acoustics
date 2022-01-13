from rest_framework import serializers
from rest_framework.serializers import FloatField


class BuoySerializer(serializers.Serializer):
    lat = FloatField()
    long = FloatField()