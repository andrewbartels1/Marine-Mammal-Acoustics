from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import AISModel
from .serializers import AISModelSerializer
from utils import date_str


@api_view(['GET'])
def boat_locations(request):
    boat_type = request.GET.get('boat_type', None)
    month = min([max([int(request.GET.get('month', 1)), 1]), 12])
    day = min([max([int(request.GET.get('day', 1)), 1]), 31])
    hour = min([max([int(request.GET.get('hour', 0)), 1]), 23])
    # min = request.GET.get('min', 0)

    filters = {
        'base_date_time__gte': date_str(month, day, hour, 0),
        'base_date_time__lt': date_str(month, day, hour, 30),
    }
    if boat_type:
        filters['boat_type'] = boat_type
    
    locs = AISModel.objects.filter(**filters).order_by('-base_date_time')

    return Response(AISModelSerializer(locs, many=True).data, status=status.HTTP_200_OK)
