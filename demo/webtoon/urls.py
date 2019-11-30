from django.urls import include, path, re_path
from . import views 

urlpatterns = [
    re_path(r'\w{1,}', views.post_list, name='post')
]