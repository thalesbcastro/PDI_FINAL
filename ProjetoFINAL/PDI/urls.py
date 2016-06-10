"""PDI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
#!/usr/bin/env python
#coding: utf8

from django.conf.urls import url, patterns
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from PDI.core_1.views import deteccao

admin.autodiscover()

urlpatterns = patterns('',
    # no caso eu so inicio e manda pra la r'^'
    # criando o namespace, posso no base.html identificar o name das minhas urls por aplicacao
    # Caso queira usar o mesmo nome em outras aplicacoes para o mesmo projeto
    # url para teste
    url(r'^form/', 'PDI.core_1.views.form', name='form'),
    url(r'^upload/', 'PDI.core_1.views.upload', name='upload'),
    url(r'^reconhecimento_facial/', 'PDI.core_1.views.reconhecimento_facial', name='reconhecimento_facial'),
    #url(r'^salvar_db/', 'PDI.core_1.views.salvar_db', name='salvar_db'),
    url(r'^olhos/', 'PDI.core_1.views.olhos', name='olhos'),
    url(r'^filtros/', 'PDI.core_1.views.filtros', name='filtros'),
    url(r'^deteccao/', 'PDI.core_1.views.deteccao', name='deteccao'),
    url(r'^admin/', admin.site.urls),
)

# Configuracao para os arquivos de media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
