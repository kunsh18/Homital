from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(UserAdmin):
    list_display = ("email", "full_name", "mobile_number", "gender", "country", "state", "is_active", "created_at")
    search_fields = ("email", "full_name", "mobile_number")
    list_filter = ("gender", "country", "is_active", "agreed_to_terms")
    ordering = ("-created_at",)
    fieldsets = UserAdmin.fieldsets + (
        ("Profile Info", {"fields": ("full_name", "mobile_number", "date_of_birth", "gender", "country", "state", "agreed_to_terms")}),
    )
