#coding: utf8
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from .models import Opencv
from .forms import FormOpencv


import numpy as np
import cv2
import PIL
from PIL import Image
#from PIL import Image
import urllib
from django.http import JsonResponse
import json
import base64
import os

# FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
#  	base_path=os.path.abspath(os.path.dirname(__file__)))
# dirname = 'core_1/static/img'
# os.chdir(dirname)
labels_global = [] # é usado em duas funcoes, modificado apenas em uma
def filtros(request):
    form = FormOpencv(request.POST or None, request.FILES or None)
    if form.is_valid():
        # class 'PDI.core_1.models.Opencv
        obj = Opencv(imagem = request.FILES['imagem'])
        print obj
        imagem = recebe_imagem(obj)

        # Processamento dos filtros com opencv e Python
        canny = cv2.Canny(imagem, 100, 200) # os segundo e terceiro argumentos são o limiar mínimo e máximo
        # filtro gaussiano
        gaussiano = cv2.GaussianBlur(imagem, (5,5), 0) # passa a imagem, o tamanho do kernel e o desvio padrão
        # filtro laplaciano
        laplaciano = cv2.Laplacian(imagem,cv2.CV_64F)
        # filtro sobel em x
        sobelx = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=5)
        # filtro sobel em y
        sobely = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=5)
        # negativo
        negativo = 255 - imagem

        cv2.imwrite("PDI/core_1/static/img/original.png", imagem)
        cv2.imwrite("PDI/core_1/static/img/canny.png", canny)
        cv2.imwrite("PDI/core_1/static/img/laplaciano.png", laplaciano)
        cv2.imwrite("PDI/core_1/static/img/gaussiano.png", gaussiano)
        cv2.imwrite("PDI/core_1/static/img/sobelx.png", sobelx)
        cv2.imwrite("PDI/core_1/static/img/sobely.png", sobely)
        cv2.imwrite("PDI/core_1/static/img/negativo.png", negativo)

        data = True
        return render(request, 'filtros.html', {'form': form, 'data': data})

    else:
        form = FormOpencv()

    return render(request, 'filtros.html', {'form': FormOpencv})


def recebe_imagem(obj):
    # imagem.extencao
    imagem_db = obj.imagem
    print imagem_db
    imagem_read = imagem_db.read() # type 'str'
    imagem_np = np.asarray(bytearray(imagem_read), dtype="uint8") # type 'numpy.ndarray'
    imagem_op = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR) # type 'numpy.ndarray'

    # retorna imagem para procesamento
    return imagem_op

def deteccao(request):

    form = FormOpencv(request.POST or None, request.FILES or None)
    if form.is_valid():
        # class 'PDI.core_1.models.Opencv
        modelo = Opencv(imagem = request.FILES['imagem'])
        imagem = recebe_imagem(modelo)

        # PROCESSAMENTO COM O OPENCV
        cv2.imwrite("PDI/core_1/static/img/deteccao_original.png", imagem) # copio pro mesmo diretorio
        # img = cv2.imread('color.png') # uso a função imread para ler a imagem num formato que cvtColor aceite
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # converto para grayScale
        vermelho = (0, 0, 255)
        # Lê arquivo .xml
        xml_face = cv2.CascadeClassifier('/home/thales/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        # detecta a face na imagem em gray
        faces = xml_face.detectMultiScale(gray, 1.3, 5)

        # retorna 4 posições
        for (x, y, w, h) in faces:
            # Ponto de saida e chegada, cor e espessura da linha
            cv2.rectangle(imagem, (x, y), (x+w, y+h), vermelho, 2)

        cv2.imwrite("PDI/core_1/static/img/deteccao.png", imagem)


        data = True
        return render(request, 'deteccao.html', {'form': form, 'data': data})

    else:
        form = FormOpencv()

    return render(request, "deteccao.html", {'form': FormOpencv})

def olhos(request):
    if request.method == 'GET':
        img = cv2.imread('PDI/core_1/static/img/deteccao_original.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converto para grayScale
        azul = (255, 0, 0)

        xml_face = cv2.CascadeClassifier('/home/thales/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
        xml_eye = cv2.CascadeClassifier('/home/thales/opencv/data/haarcascades/haarcascade_eye.xml')

        faces = xml_face.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = xml_eye.detectMultiScale(roi_gray)
            for (x, y, w, h) in eyes:
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), azul, 2)

        cv2.imwrite('PDI/core_1/static/img/detect_olho.png', img)

    return render(request, 'eye.html', {})


def reconhecimento_facial(request):
    form = FormOpencv(request.POST or None, request.FILES or None)
    if form.is_valid():
        obj = Opencv(imagem = request.FILES['imagem'])

        # Imagem preparada
        # LEMBRAR QUE DEVE SER GLOBAL PARA PODER SER ACESSADA DENTRO DA FUNÇÃO
        # form e upload, CASO O USUÁRIO DESEJE SALVAR
        img = recebe_imagem(obj)

        #Retornar um booleano

        booleano = reconhecimento(img)

        if booleano:
            # Preciso passar Imagem original, imagem compativel
            '''
                Faco isso na funcao reconhecimento.
                if imagem compativel:
                    cv2.imread('PDI/core_1/static/img/imagem_original', img orginal)
                    cv2.imread('PDI/core_1/static/img/imagem_reconhecida', img compativel)
            '''
            # Entao vou poder acessar a imagem pelo nome dentro do template reconhecimento_facial.html
            #print "Passou"
            data = True
            return render(request, "fim.html", {'data': data})
        # retorno salvar.html
        else:
            # Em salvar.html deve ter a mensagem informando que nao existe e perguntando se deseja salvar
            print "Deseja salvar??"
            data = False
            return render(request, "fim.html", {'data': data})
    else:
        form = FormOpencv()

    return render(request, "reconhecimento_facial_form.html", {'form':FormOpencv})

# Caso exista uma requisação para salvar a imagem
def form(request):
    return render(request, "salvar.html", {})

def upload(request):
    global labels_global
    #print labels_global
    path = '/home/thales/Documentos/ProjetoTeste/PDI/imagens/'
    # Caso a lista de labels esteja vazia
    # #global labels
    # if not labels_global:
    #     labels_global = [15]
    # max_labels = max(labels_global)
    lista_ext = ['.happy', '.surprised', '.normal', '.centerlight', '.wink']
    j = 0
    for count, x in enumerate(request.FILES.getlist("files")):
        #name_ant = "subject" + num
        #name = str(max_labels+1) + lista_ext[j]
        '''
            Como eu tou fazendo primeiro ele passar por essa funcao ao inves da "reconhecimento", meu 'labels' será smp vazio, entao o fazer de
            forma automatica nao vai dar, logo fixo um numero, no caso 03, pros nomes das minhas imagens baixadas
        '''
        def process(f):
            with open('/home/thales/Documentos/ProjetoTeste/PDI/imagens/subject03' + lista_ext[j], 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
        process(x)
        j = j + 1
    images = [os.path.join(path, f) for f in os.listdir(path)]
    print images
    '''
        Mando pro 'salvo_com_sucesso.html', que de lá mandará para funcão 'reconhecimento_facial' que abrirá um formularioa
        para baixar a imagem a ser testada. Em seguida a imagem é testada na views 'reconhecimento' que retornará um true ou false

    '''
    return render(request, "salvo_com_sucesso.html", {})
# urls:
# reconhecimento_facial, salvar_db


# LEMBRAR DE PENSAR MAIS SOBRE O 'LABELS'
def reconhecimento(img):
    path = '/home/thales/Documentos/ProjetoTeste/PDI/imagens/'
    xml_face = cv2.CascadeClassifier('/home/thales/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

    recognizer = cv2.face.createLBPHFaceRecognizer()
    # adiciono tds os caminhos a lista 'image_paths' com exceção da minha imagem .sad
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    images = []
    labels = []

    # para todas as imagens com exceção da que será testada
    for image_path in image_paths:
        # lê imagem e converte pra grayscale
        image_pil = Image.open(image_path).convert('L')
        # converte imagem para formato numpy array
        image_np = np.array(image_pil, 'uint8')
        # Pega o label da imagem
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        #Detecta a face na imagem
        faces = xml_face.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            # adiciono a face detectada da imagem a lista 'images'
            images.append(image_np[y: y + h, x: x + w])
            labels.append(nbr)
    global labels_global
    labels_global = labels
    # Realizo o treinamento com as imagens adicionadas a lista 'images'
    # o algoritomo de treinamento necessita das imagens e de lables atribuidos a cada imagem
    recognizer.train(images, np.array(labels))

    # Agora adiciono a imagem baixada a lista image_paths

    # Salvo imagem no formato png
    # adciono um numero que nao existe com o .sad
    '''
        Tenho que mandar escrever no diretório com o mesmo label (03) das imagens salvas, pois o algoritmo de reconhecimento usado
        pelo opencv utiliza as características da imagem, bem como o seu label para um perfeito reconhecimento.
    '''

    cv2.imwrite('PDI/tmp/subject03.sad.png', img)
    image_test = Image.open('PDI/tmp/subject03.sad.png').convert('L')
    image_test.save('PDI/imagens/subject03.sad', 'gif')



    # Adiciono a imagem so com o .sad
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
    # converto a imagem para grayscale
    for image_path in image_paths:
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
    # converto num numpy
    #image_test_np = np.array(image_test, 'uint8')
    # Detecto a face
        faces = xml_face.detectMultiScale(predict_image)
        for (x, y, z, h) in faces:

            nbr_test, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            print conf
            nbr_das_imagens_treinadas = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            if  conf <= 50:#(nbr_test == nbr_das_imagens_treinadas):
                print conf
                img = cv2.cvtColor(predict_image, cv2.COLOR_GRAY2RGB)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imwrite('PDI/core_1/static/img/imagemRec.png', img)

                return True
            else:
                 print "Passou"
                 return False
