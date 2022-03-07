from django.db import models

# Create your models here.

class Teeth_Model(models.Model):

    image_path = models.ImageField(upload_to = 'images/')
    detected_image = models.ImageField(upload_to = 'imagesresult/')
    teeth_score = models.TextField()

    def delete(self,*args,**kwargs):
        self.image.delete()
        self.detected_img.delete()
        super().delete(*args,**kwargs)
