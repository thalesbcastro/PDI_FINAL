#coding: utf8
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Opencv(models.Model):
    imagem = models.ImageField(upload_to="opencv/media")
