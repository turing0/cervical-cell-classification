from django.shortcuts import render, redirect
from web import models
from cervical_cell_classification import settings
from .myforms import UploadFileForm
from .models import User
from django.http import HttpResponse
from celery_tasks.sms.tasks import pred
from celery_tasks.sms.tasks import predict_churn_single
from .functions import predict

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
        # print(form)
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
            # datalist = handle_uploaded_file(request.FILES['file'],filename=request.FILES.get('file'))
            # message = 'Done! 已处理 '
            # valueList = datalist[1::2]
            # tlist = apriorirules(safe_path)
            # reslist = tlist[0]
            # xlist = tlist[1]

            path = settings.MEDIA_ROOT + '/' + str(user.headimg)
            # print(path)

            # result = add.apply_async(args=[3, 5])
            #
            # outputLabel, predicted = predict(path)
            # outputLabel, predicted, datas = pred(path, user.headimg.path)

            # result = pred.apply_async(args=[path, user.headimg.path])
            result = predict_churn_single.apply_async(args=[path, user.headimg.path, str(user.headimg)])

            return render(request, 'system.html', locals())
    else:
        form = UploadFileForm()

    return render(request, 'system.html', locals())


def tutorial(request):
    return render(request, 'tutorial.html', locals())


def intro(request):
    return render(request, 'intro.html', locals())


def data(request):
    return render(request, 'data.html', locals())


def index(request):
    return render(request, 'index.html', locals())


def job(request):
    from celery.result import AsyncResult
    if request.method == 'POST':
        id = request.POST.get('id').strip()
        if id:
            status = AsyncResult(id).status
            result = AsyncResult(id).result
            origin_image = result['origin_image']
            segmentation_image = result['segmentation_image']
            predicted = result['predicted']
            nuclear_area = result['shape_features'][0]
            cell_area = result['shape_features'][1]
            R_mean = result['color_features'][0]
            G_mean = result['color_features'][1]
            B_mean = result['color_features'][2]
            R_variance = result['color_features'][3]
            G_variance = result['color_features'][4]
            B_variance = result['color_features'][5]
            energy = result['texture_features'][0]
            asm = result['texture_features'][1]
            contrast = result['texture_features'][2]
            correlation = result['texture_features'][3]
            return render(request, 'job.html', locals())
    return render(request, 'job.html', locals())


def details(request):
    return render(request, 'details.html', locals())
