from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
from .utils import predict_bark

@csrf_exempt
def predict_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        file_path = default_storage.save(img_file.name, img_file)

        full_path = os.path.join(default_storage.location, file_path)
        result = predict_bark(full_path)

        os.remove(full_path)  # Clean up uploaded file

        return JsonResponse(result)

    return JsonResponse({'error': 'Please upload an image file.'}, status=400)

# Create your views here.
