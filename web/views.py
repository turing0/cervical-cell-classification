from django.shortcuts import render, redirect
from web import models
from .myforms import UploadFileForm
from .models import User
from celery_tasks.sms.tasks import predict_churn_multiple
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
            image_list = []
            for file in request.FILES.getlist('file'):

                if str(file).find('.BMP') == -1:
                    print('skip ', file)
                    continue
                # print(file)
                fn = file
                user = User(headimg=file)
                user.save()
                print('---------------')
                print(user.headimg)
                print(user.headimg.path)
                image_list.append([user.headimg.path, str(user.headimg)])

            # result = predict_churn_single.apply_async(args=[path, user.headimg.path, str(user.headimg)])
            result = predict_churn_multiple.apply_async(args=[image_list])

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

from django_celery_results.models import TaskResult

def job(request):
    from celery.result import AsyncResult
    if request.method == 'POST':
        id = request.POST.get('id').strip()
        if id:
            return_list = []

            status = AsyncResult(id).status
            result_list = AsyncResult(id).result

            for i, result in enumerate(result_list):
                temp_dict = dict()
                temp_dict['origin_image'] = result['origin_image']
                temp_dict['segmentation_image'] = result['segmentation_image']
                temp_dict['predicted'] = result['predicted']
                temp_dict['nuclear_area'] = result['shape_features'][0]
                temp_dict['cell_area'] = result['shape_features'][1]
                temp_dict['R_mean'] = result['color_features'][0]
                temp_dict['G_mean'] = result['color_features'][1]
                temp_dict['B_mean'] = result['color_features'][2]
                temp_dict['R_variance'] = result['color_features'][3]
                temp_dict['G_variance'] = result['color_features'][4]
                temp_dict['B_variance'] = result['color_features'][5]
                temp_dict['energy'] = result['texture_features'][0]
                temp_dict['asm'] = result['texture_features'][1]
                temp_dict['contrast'] = result['texture_features'][2]
                temp_dict['correlation'] = result['texture_features'][3]
                temp_dict['idx'] = i + 1
                return_list.append(temp_dict)
            counts = len(return_list)
            return render(request, 'job.html', locals())
    else:
        results = TaskResult.objects.all()
        task_results_list = list(results.values())  # 将QuerySet转换为列表
        print(task_results_list)
        jobId_list = [{0:item['task_id'], 1:item['date_created']} for item in task_results_list]
    return render(request, 'job.html', locals())


def details(request):
    return render(request, 'details.html', locals())
