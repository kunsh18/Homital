from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, LoginForm


def register(request):
    """Handle user registration."""
    if request.user.is_authenticated:
        return redirect("user:dashboard")

    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f"🎉 Account created successfully! Welcome, {user.full_name}! You can now log in."
            )
            return redirect("user:register_success")
        else:
            messages.error(request, "Please correct the errors below and try again.")
    else:
        form = UserRegistrationForm()

    return render(request, "user/register.html", {"form": form})


def register_success(request):
    """Registration success confirmation page."""
    return render(request, "user/register_success.html")


def login_view(request):
    """Handle user login with email + password."""
    if request.user.is_authenticated:
        return redirect("user:dashboard")

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            login(request, user)
            messages.success(request, f"Welcome back, {user.full_name}! 👋")
            # Redirect to 'next' param if present, otherwise dashboard
            next_url = request.GET.get("next", "user:dashboard")
            return redirect(next_url)
        else:
            messages.error(request, "Login failed. Please check your credentials.")
    else:
        form = LoginForm()

    return render(request, "user/login.html", {"form": form})


def logout_view(request):
    """Log the user out and redirect to login page."""
    if request.method == "POST":
        logout(request)
        messages.success(request, "You have been logged out successfully.")
    return redirect("user:login")


@login_required(login_url="user:login")
def dashboard(request):
    """Basic dashboard shown after successful login."""
    return render(request, "user/dashboard.html", {"user": request.user})
