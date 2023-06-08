from django.contrib.auth import login, authenticate
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect

from .model import Text
from .similarity import similarity_checker


def index(request):
    context = {}
    if request.method == 'POST':
        text1 = request.POST.get('text1')
        text2 = request.POST.get('text2')

        prediction, probability = similarity_checker.check_similarity(text1, text2)
        context['result'] = f"Prediction: {prediction}, Probability: {probability}"

        context['text1'] = text1
        context['text2'] = text2

        if request.user.is_authenticated:
            text = Text(user=request.user, text1=text1, text2=text2,
                        result=context['result'])
            text.save()

    return render(request, 'index.html', context)


def logout_view(request):
    logout(request)
    return redirect('index')


def signup_view(request):
    error = None
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('index')
        else:
            error = "Incorrect registration data"
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form, 'error': error})


def login_view(request):
    context = {'form': AuthenticationForm()}
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
        else:
            context['form'] = form
            context['error'] = 'Incorrect password or username'
    return render(request, 'login.html', context)


@login_required
def my_texts_view(request):
    texts = Text.objects.filter(user=request.user)
    return render(request, 'my_texts.html', {'texts': texts})
