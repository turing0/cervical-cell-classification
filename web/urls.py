from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='intro'),
    path(r'index/', views.index, name='index'),
    path(r'system/', views.system, name='system'),
    path(r'tutorial/', views.tutorial, name='tutorial'),
    path(r'contact/', views.contact, name='contact'),
    path(r'job/', views.job, name='job'),
    path(r'tutorial/blog_single/', views.blog_single, name='blog_single'),

    # path(r'history/', views.historydata, name='historydata'),
    # path(r'^share$', views.share, name='share'),
    # path('gethtml/', views.gethtml),

]