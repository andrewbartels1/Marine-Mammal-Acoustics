from rest_framework import serializers
from pams.models import AudioClip


class AudioClipSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioClip
        fields = '__all__'