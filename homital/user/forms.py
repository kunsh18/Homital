from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.contrib.auth import authenticate
from .models import UserProfile, GENDER_CHOICES

import re
from datetime import date


COUNTRY_STATE_CHOICES = [
    ("", "-- Select Country/State --"),
    # India
    ("IN-AN", "India - Andaman and Nicobar Islands"),
    ("IN-AP", "India - Andhra Pradesh"),
    ("IN-AR", "India - Arunachal Pradesh"),
    ("IN-AS", "India - Assam"),
    ("IN-BR", "India - Bihar"),
    ("IN-CH", "India - Chandigarh"),
    ("IN-CG", "India - Chhattisgarh"),
    ("IN-DN", "India - Dadra and Nagar Haveli"),
    ("IN-DD", "India - Daman and Diu"),
    ("IN-DL", "India - Delhi"),
    ("IN-GA", "India - Goa"),
    ("IN-GJ", "India - Gujarat"),
    ("IN-HR", "India - Haryana"),
    ("IN-HP", "India - Himachal Pradesh"),
    ("IN-JK", "India - Jammu and Kashmir"),
    ("IN-JH", "India - Jharkhand"),
    ("IN-KA", "India - Karnataka"),
    ("IN-KL", "India - Kerala"),
    ("IN-LA", "India - Ladakh"),
    ("IN-LD", "India - Lakshadweep"),
    ("IN-MP", "India - Madhya Pradesh"),
    ("IN-MH", "India - Maharashtra"),
    ("IN-MN", "India - Manipur"),
    ("IN-ML", "India - Meghalaya"),
    ("IN-MZ", "India - Mizoram"),
    ("IN-NL", "India - Nagaland"),
    ("IN-OD", "India - Odisha"),
    ("IN-PY", "India - Puducherry"),
    ("IN-PB", "India - Punjab"),
    ("IN-RJ", "India - Rajasthan"),
    ("IN-SK", "India - Sikkim"),
    ("IN-TN", "India - Tamil Nadu"),
    ("IN-TS", "India - Telangana"),
    ("IN-TR", "India - Tripura"),
    ("IN-UP", "India - Uttar Pradesh"),
    ("IN-UK", "India - Uttarakhand"),
    ("IN-WB", "India - West Bengal"),
    # United States
    ("US-AL", "USA - Alabama"),
    ("US-AK", "USA - Alaska"),
    ("US-AZ", "USA - Arizona"),
    ("US-CA", "USA - California"),
    ("US-CO", "USA - Colorado"),
    ("US-FL", "USA - Florida"),
    ("US-GA", "USA - Georgia"),
    ("US-HI", "USA - Hawaii"),
    ("US-IL", "USA - Illinois"),
    ("US-NY", "USA - New York"),
    ("US-TX", "USA - Texas"),
    ("US-WA", "USA - Washington"),
    # UK
    ("GB-ENG", "United Kingdom - England"),
    ("GB-NIR", "United Kingdom - Northern Ireland"),
    ("GB-SCT", "United Kingdom - Scotland"),
    ("GB-WLS", "United Kingdom - Wales"),
    # Other countries (no states listed)
    ("AU", "Australia"),
    ("CA", "Canada"),
    ("DE", "Germany"),
    ("FR", "France"),
    ("JP", "Japan"),
    ("SG", "Singapore"),
    ("AE", "United Arab Emirates"),
    ("OTHER", "Other"),
]


class UserRegistrationForm(forms.ModelForm):
    """Full user registration form with all required fields."""

    full_name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={"placeholder": "Enter your full name", "id": "id_full_name"}),
        label="Full Name",
    )

    email = forms.EmailField(
        widget=forms.EmailInput(attrs={"placeholder": "Enter your email address", "id": "id_email"}),
        label="Email Address",
    )

    mobile_number = forms.CharField(
        max_length=15,
        widget=forms.TextInput(attrs={"placeholder": "+91 9876543210", "id": "id_mobile_number"}),
        label="Mobile Number",
    )

    password = forms.CharField(
        min_length=8,
        widget=forms.PasswordInput(attrs={"placeholder": "Min. 8 characters", "id": "id_password"}),
        label="Password",
    )

    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "Re-enter your password", "id": "id_confirm_password"}),
        label="Confirm Password",
    )

    date_of_birth = forms.DateField(
        widget=forms.DateInput(attrs={"type": "date", "id": "id_date_of_birth"}),
        label="Date of Birth",
    )

    gender = forms.ChoiceField(
        choices=[("", "-- Select Gender --")] + list(GENDER_CHOICES),
        widget=forms.Select(attrs={"id": "id_gender"}),
        label="Gender",
    )

    country_state = forms.ChoiceField(
        choices=COUNTRY_STATE_CHOICES,
        widget=forms.Select(attrs={"id": "id_country_state"}),
        label="Country / State",
    )

    agreed_to_terms = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={"id": "id_agreed_to_terms"}),
        label="I agree to the Terms of Service and Privacy Policy",
        error_messages={"required": "You must agree to the Terms of Service and Privacy Policy to register."},
    )

    class Meta:
        model = UserProfile
        fields = [
            "full_name",
            "email",
            "mobile_number",
            "date_of_birth",
            "gender",
            "agreed_to_terms",
        ]

    # ----- Field-level validation -----

    def clean_full_name(self):
        name = self.cleaned_data.get("full_name", "").strip()
        if len(name) < 2:
            raise ValidationError("Full name must be at least 2 characters.")
        return name

    def clean_mobile_number(self):
        mobile = self.cleaned_data.get("mobile_number", "").strip()
        # Allow digits, spaces, dashes, parentheses, and optional leading +
        if not re.match(r"^\+?[\d\s\-\(\)]{7,15}$", mobile):
            raise ValidationError("Enter a valid mobile number (7–15 digits, optionally starting with +).")
        return mobile

    def clean_email(self):
        email = self.cleaned_data.get("email", "").lower()
        if UserProfile.objects.filter(email=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def clean_date_of_birth(self):
        dob = self.cleaned_data.get("date_of_birth")
        if dob:
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if age < 13:
                raise ValidationError("You must be at least 13 years old to register.")
            if age > 120:
                raise ValidationError("Please enter a valid date of birth.")
        return dob

    def clean_password(self):
        password = self.cleaned_data.get("password", "")
        if not re.search(r"[A-Z]", password):
            raise ValidationError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", password):
            raise ValidationError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", password):
            raise ValidationError("Password must contain at least one digit.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            raise ValidationError("Password must contain at least one special character.")
        return password

    # ----- Cross-field validation -----

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            self.add_error("confirm_password", "Passwords do not match.")

        # Split country_state into country and state fields
        country_state = cleaned_data.get("country_state", "")
        if "-" in country_state:
            parts = country_state.split("-", 1)
            cleaned_data["country"] = parts[0]
            cleaned_data["state"] = parts[1]
        else:
            cleaned_data["country"] = country_state
            cleaned_data["state"] = ""

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])

        # Auto-generate a unique username from email
        email = self.cleaned_data["email"]
        base_username = email.split("@")[0]
        username = base_username
        counter = 1
        while UserProfile.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1
        user.username = username

        user.country = self.cleaned_data.get("country", "")
        user.state = self.cleaned_data.get("state", "")

        if commit:
            user.save()
        return user


class LoginForm(forms.Form):
    """Login form — authenticates using email + password."""

    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            "placeholder": "Enter your email address",
            "id": "id_login_email",
            "autocomplete": "email",
        }),
        label="Email Address",
    )

    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            "placeholder": "Enter your password",
            "id": "id_login_password",
            "autocomplete": "current-password",
        }),
        label="Password",
    )

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get("email", "").lower()
        password = cleaned_data.get("password", "")

        if email and password:
            user = authenticate(username=email, password=password)
            if user is None:
                raise forms.ValidationError(
                    "Invalid email or password. Please try again."
                )
            if not user.is_active:
                raise forms.ValidationError(
                    "This account has been deactivated. Please contact support."
                )
            cleaned_data["user"] = user

        return cleaned_data
