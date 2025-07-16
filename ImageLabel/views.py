from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from urllib3 import request
from .serializers import ImageUploadSerializer
from .models import predict_label_from_image

class PredictImageView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        print(request.data)
        print(request.FILES)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            predictions = predict_label_from_image(image_file)
            return Response({"predictions": predictions})
        return Response(serializer.errors, status=400)
