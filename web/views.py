from django.shortcuts import render, redirect
from web import models
from cervical_cell_classification import settings
from .myforms import UploadFileForm
from .models import User
from .functions import compute_shape_features
from .functions import compute_color_features
from .functions import compute_texture_features
from .functions import predict
import os

res = {
    '0': '原位癌细胞',
    '1': '重度鳞状异常细胞',
    '2': '正常上皮鳞状细胞',
    '3': '低度鳞状异常细胞',
    '4': '正常中层鳞状细胞',
    '5': '正常柱状细胞',
    '6': '中度鳞状异常细胞',
}


def system(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print(form)
        if form.is_valid():
            # file = time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '_' + str(request.FILES.get('file'))
            file = request.FILES.get('file')
            # headimg = form.cleaned_data['file']

            if str(file).find('.BMP') == -1:
                message = '请上传 BMP 文件！'
                return render(request, 'system.html', locals())
            fn = request.FILES['file']
            user = User(headimg=file)
            user.save()

            print('---------------')
            print(user.headimg)

            nuclear_area, cell_area = compute_shape_features('output_img.jpg')
            R_mean, G_mean, B_mean, R_variance, G_variance, B_variance = compute_color_features(user.headimg.path)
            energy, contrast, asm, correlation = compute_texture_features(user.headimg.path)
            # datalist = handle_uploaded_file(request.FILES['file'],filename=request.FILES.get('file'))
            # message = 'Done! 已处理 '
            # valueList = datalist[1::2]
            # tlist = apriorirules(safe_path)
            # reslist = tlist[0]
            # xlist = tlist[1]

            # print(settings.MEDIA_ROOT)
            # path = os.path.join(settings.MEDIA_ROOT, str(user.headimg))
            path = settings.MEDIA_ROOT + '/' + str(user.headimg)
            # print(path)
            outputLabel, predicted = predict(path)

            predicted = res[str(predicted)[8]]
            print(outputLabel, predicted)

            return render(request, 'system.html', locals())
    else:
        form = UploadFileForm()

    return render(request, 'system.html', locals())


# 0   carcinoma_in_situ   原位癌细胞
# 1   severe_dysplastic   重度鳞状异常细胞
# 2   normal_superficiel  正常上皮鳞状细胞
# 3   light_dysplastic    低度鳞状异常细胞
# 4   normal_intermediate 正常中层鳞状细胞
# 5   normal_columnar     正常柱状细胞
# 6   moderate_dysplastic 中度鳞状异常细胞

def tutorial(request):
    return render(request, 'tutorial.html', locals())


def contact(request):
    return render(request, 'contact.html', locals())


def intro(request):
    return render(request, 'intro.html', locals())


def data(request):
    return render(request, 'data.html', locals())


def index(request):
    return render(request, 'index.html', locals())

def job(request):
    return render(request, 'job.html', locals())