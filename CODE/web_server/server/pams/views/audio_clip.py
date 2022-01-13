import os
from django.http import FileResponse
from django.db.models import Q
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pams.models import AudioClip
from pams.serializers import AudioClipSerializer, BuoySerializer
from utils import date_str


@api_view(['GET'])
def buoy_locations(request):
    buoys = AudioClip.objects.raw('select 0 as id, * from (select distinct lat, long from pams_audioclip)')
    serializer = BuoySerializer(buoys, many=True)

    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
def audio_file(request):
    id = request.GET.get('id', None)
    label = request.GET.get('label', None)
    
    try:
        filename = f'/app/pams/static/audio/{label}/{id}/chip.'
        os.system(f'ffmpeg -i {filename}wav {filename}mp3 -y')
        audio = open(f'/app/pams/static/audio/{label}/{id}/chip.mp3', 'rb')
        response = FileResponse(audio)
        response['Content-Type'] = 'audio/mp3'

        return response
    except:
        pass

    return Response({}, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
def audio_clips(request):
    month = min([max([int(request.GET.get('month', 1)), 1]), 12])
    day = min([max([int(request.GET.get('day', 1)), 1]), 31])
    hour = min([max([int(request.GET.get('hour', 0)), 1]), 23])
    lat = request.GET.get('lat', None)
    long = request.GET.get('long', None)
    label = request.GET.get('classification', None)
    page = request.GET.get('page', 1)
    page_size = 10000

    filters = {
        'chip_start_seconds__gte': date_str(month, day, hour, 0),
        'chip_start_seconds__lt': date_str(month, day, hour, 59),
    }
    if lat:
        filters['lat'] = lat
    if long:
        filters['long'] = long

    clips = AudioClip.objects.filter(**filters)
    if label:
        clips = clips.filter(~Q(label='unknown')) if label == 'unknown' else clips.filter(label=label)

    serializer = AudioClipSerializer(clips[page_size*(page-1):page_size*(page)], many=True)

    return Response(serializer.data, status=status.HTTP_200_OK)
