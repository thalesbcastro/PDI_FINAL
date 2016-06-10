
from django.forms import ModelForm
from .models import Opencv

class FormOpencv(ModelForm):
    class Meta:
        model = Opencv
        fields = ['imagem']
