from django.db import models
from django.contrib.auth.models import User


class Text(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text1 = models.TextField()
    text2 = models.TextField()
    result = models.CharField(max_length=200)
