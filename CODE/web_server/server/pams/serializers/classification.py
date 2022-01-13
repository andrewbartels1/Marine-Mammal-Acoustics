from rest_framework import serializers
from rest_framework.serializers import CharField


class ClassificationSerializer(serializers.Serializer):
    name = CharField()
    filename = CharField()