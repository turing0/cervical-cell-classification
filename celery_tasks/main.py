from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cervical_cell_classification.settings')

app = Celery('cervical_cell_classification', backend="redis://127.0.0.1/10")

app.config_from_object('celery_tasks.config', namespace='CELERY')

# 自动从Django的已注册app中发现任务
app.autodiscover_tasks()
