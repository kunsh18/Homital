from django.urls import path
from . import views

app_name = "user"

urlpatterns = [
    path("register/", views.register, name="register"),
    path("register/success/", views.register_success, name="register_success"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
