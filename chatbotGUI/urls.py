from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # 기본 경로를 index 뷰로 설정
    path('chat_response_post/', views.chat_response_post, name='chat_response_post'),
    path('chat_response_stream/', views.chat_response_stream, name='chat_response_stream'),
]
