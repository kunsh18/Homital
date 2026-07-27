from django.db import models
from django.contrib.auth.models import AbstractUser


GENDER_CHOICES = [
    ("M", "Male"),
    ("F", "Female"),
    ("O", "Other"),
    ("P", "Prefer not to say"),
]


class UserProfile(AbstractUser):
    """Extended user model with additional profile fields."""

    # Override first/last name with full name
    full_name = models.CharField(max_length=255)

    # Contact
    mobile_number = models.CharField(max_length=15, unique=True)

    # Personal details
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, blank=True)

    # Location
    country = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)

    # Legal consent
    agreed_to_terms = models.BooleanField(default=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Override email from AbstractUser to enforce uniqueness (required for USERNAME_FIELD)
    email = models.EmailField(unique=True)

    # Use email as the username field for login
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username", "full_name", "mobile_number"]

    def __str__(self):
        return f"{self.full_name} ({self.email})"

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"
