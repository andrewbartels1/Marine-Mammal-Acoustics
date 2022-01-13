from rest_framework import serializers
from .models import AISModel


class AISModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = AISModel
        fields = '__all__'