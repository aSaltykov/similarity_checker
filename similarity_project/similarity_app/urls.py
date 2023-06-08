from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('mytexts/', views.my_texts_view, name='mytexts'),
    path('logout/', views.logout_view, name='logout'),
]
