from django.shortcuts import render
import numpy as np
from . import function
import cv2
import os
from django.http import HttpResponseRedirect,HttpResponse
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from main_app.forms import ImageForm
from django.views.generic import DetailView
from main_app.models import Teeth_Model
import tensorflow as tf
from PIL import Image as Img
# Create your views here.

class Image(TemplateView):
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(base_dir, 'model.tflite')
    #
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    tr = 0.3
    form = ImageForm
    template_name = 'image.html'

    def post(self, request, *args, **kwargs):

        form = ImageForm(request.POST, request.FILES)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'model.tflite')

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        tr = 0.3


        if form.is_valid():
            obj = form.save()

            detection_result_image = function.run_odt_and_draw_results(
                obj.image_path.path,
                interpreter,
                threshold=tr
            )

            output_image = Img.fromarray(detection_result_image)
            start_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_dir = os.path.join(start_dir, 'media/imagesresult')
            output_image.save(os.path.join(file_dir, 'abc.jpg'))

            obj.detected_image = os.path.join(file_dir, 'abc.jpg')

            detection_result_data = function.show_result_table(
                interpreter,
                obj.image_path.path,
                tr
            )

            out = function.final_result(detection_result_data)

            #outputFile=os.path.basename(outputFile)
            obj.teeth_score = out
            obj.save()

            return HttpResponseRedirect(reverse_lazy('image_display', kwargs={'pk': obj.id}))

        context = self.get_context_data(form=form)
        return self.render_to_response(context)

    def get(self, request, *args, **kwargs):
        return self.post(request, *args, **kwargs)

class ImageDisplay(DetailView):
    model = Teeth_Model
    template_name = 'image_display.html'
    context_object_name = 'context'

def deleteimg(request,pk):
    if request.method=='POST':
        # model = BMI_Model.objects.get(pk=pk)
        # model.delete()
        return HttpResponseRedirect(reverse_lazy('home'))
