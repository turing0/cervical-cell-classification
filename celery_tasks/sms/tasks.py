from celery_tasks.main import app
from celery import Task
from celery import shared_task
import time
from web.functions import predict
from web.functions import compute_shape_features
from web.functions import compute_color_features
from web.functions import compute_texture_features
import importlib

res = {
    '0': '原位癌细胞',
    '1': '重度鳞状异常细胞',
    '2': '正常上皮鳞状细胞',
    '3': '低度鳞状异常细胞',
    '4': '正常中层鳞状细胞',
    '5': '正常柱状细胞',
    '6': '中度鳞状异常细胞',
}


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.model:
            print('Loading Model...')
            module_import = importlib.import_module(self.path[0])
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj()
            print('Model loaded')
        return self.run(*args, **kwargs)


@app.task(ignore_result=False,
          bind=True,
          base=PredictTask,
          path=('web.model', 'ChurnModel'),
          name='{}.{}'.format(__name__, 'Churn'))
def predict_churn_single(self, file, user_headimg_path, user_headimg):
    output_label, predicted, segmentation_image = self.model.predict(file, user_headimg)
    predicted = res[str(predicted)[8]]
    print('user_headimg: ', user_headimg)
    nuclear_area, cell_area = compute_shape_features('output_images/' + 'output_img.jpg')
    R_mean, G_mean, B_mean, R_variance, G_variance, B_variance = compute_color_features(user_headimg_path)
    energy, contrast, asm, correlation = compute_texture_features(user_headimg_path)
    datas = {
        'origin_image': user_headimg,
        'segmentation_image': segmentation_image,
        'predicted': predicted,
        'shape_features': [nuclear_area, cell_area],
        'color_features': [R_mean, G_mean, B_mean, R_variance, G_variance, B_variance],
        'texture_features': [energy, contrast, asm, correlation],
    }

    print('------', predicted)
    return datas


def predict_churn_multiple(self, file, user_headimg_path, user_headimg):
    data_list = []
    data_list.append(predict_churn_single(file, user_headimg_path, user_headimg))

    return data_list


@app.task
def pred(path, user_headimg_path):
    outputLabel, predicted = predict(path)
    predicted = res[str(predicted)[8]]
    nuclear_area, cell_area = compute_shape_features('output_images/' + 'output_img.jpg')
    R_mean, G_mean, B_mean, R_variance, G_variance, B_variance = compute_color_features(user_headimg_path)
    energy, contrast, asm, correlation = compute_texture_features(user_headimg_path)
    datas = {
        'shape_features': [nuclear_area, cell_area],
        'color_features': [R_mean, G_mean, B_mean, R_variance, G_variance, B_variance],
        'texture_features': [energy, contrast, asm, correlation],
    }
    return outputLabel, predicted, datas


