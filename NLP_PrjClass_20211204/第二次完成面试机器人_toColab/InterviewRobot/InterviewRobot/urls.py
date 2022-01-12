"""InterviewRobot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
'''
urls.py
'''
from django.contrib import admin
from django.urls import path
import sys
sys.path.append('./')
from run1 import InterviewRobot,pdf2txt_server,key_word_show,get_question,send_answer


urlpatterns = [
    path('InterviewRobot', InterviewRobot),#总界面渲染后前端呈现
    path('pdf2txt/', pdf2txt_server),#pdf转txt前端选择文件路径，后端进行处理
    path('key_word_show/', key_word_show),
    path('get_question/', get_question), #question呈现到前端
    path('send_answer/',send_answer),#从前端传递面试者答案到后端，计算分数后返回前端进行呈现
]
