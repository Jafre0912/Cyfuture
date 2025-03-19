from django.urls import path
from .views import check_plagiarism

urlpatterns = [
    path('check/', check_plagiarism, name='check_plagiarism'),
]
