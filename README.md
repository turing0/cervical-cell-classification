# cervical-cell-classification
Cervical Cell Classification



# How to run it?

```bash
# 1: redis, in project root directory.
redis-server.exe

# 2: then activate celery, in your redis directory
Celery -A celery_tasks.main worker -l info -P eventlet

# 3. run the server
python manage.py runserver

```









