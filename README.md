# cervical-cell-classification
Cervical Cell Classification



# How to run it?

```bash
# 1: redis, in your redis directory
redis-server.exe

# 2: then activate celery, in project root directory
Celery -A celery_tasks.main worker -l info -P eventlet

# 3. run the server
python manage.py runserver

```









