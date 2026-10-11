import os
import re
from functools import wraps
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from utils.preprocessing import (
    validate_and_preprocess_input,
    get_crop_profile,
    FEATURE_COLUMNS,
    ALLOWED_SEASONS,
)
from utils.database import (
    init_db,
    create_user,
    get_user_by_email,
    get_user_by_id,
    save_prediction,
    get_predictions_by_user,
    get_prediction_by_id,
    get_user_prediction_count,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartcrop-ai-secret-key-2026")

# Automatically initialize SQLite database on application startup
init_db()

MODEL_PATH = os.path.join("model", "crop_model.pkl")
METADATA_PATH = os.path.join("model", "model_metadata.pkl")

# Global model cache loaded on demand
_cached_model = None
_cached_metadata = None
_cached_importances = None

FEATURE_LABELS = {
    "rainfall": "Rainfall (mm)",
    "humidity": "Relative Humidity (%)",
    "K": "Potassium (K)",
    "P": "Phosphorus (P)",
    "N": "Nitrogen (N)",
    "temperature": "Temperature (°C)",
}


@app.context_processor
def inject_user():
    """Injects current logged-in user into all templates for session-aware UI."""
    if "user_id" in session:
        return {
            "current_user": {
                "id": session.get("user_id"),
                "name": session.get("user_name"),
                "email": session.get("user_email"),
            }
        }
    return {"current_user": None}


def login_required(f):
    """Decorator ensuring routes are only accessible to authenticated farmers."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def get_model():
    """Loads and caches the trained ML model and actual feature importances."""
    global _cached_model, _cached_metadata, _cached_importances
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            return None, None, None
        try:
            _cached_model = joblib.load(MODEL_PATH)
            if os.path.exists(METADATA_PATH):
                _cached_metadata = joblib.load(METADATA_PATH)

            # Compute actual feature importances directly from Random Forest model
            if hasattr(_cached_model, "feature_importances_"):
                _cached_importances = []
                for feat, imp in zip(FEATURE_COLUMNS, _cached_model.feature_importances_):
                    _cached_importances.append({
                        "feature": feat,
                        "label": FEATURE_LABELS.get(feat, feat),
                        "percentage": round(float(imp) * 100, 2),
                        "weight": round(float(imp), 4),
                    })
                # Order by descending importance
                _cached_importances.sort(key=lambda x: x["percentage"], reverse=True)
        except Exception:
            return None, None, None
    return _cached_model, _cached_metadata, _cached_importances


# -----------------------------------------------------------------------------
# Core Prediction Routes (Unchanged ML behavior, with optional DB persistence)
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Home page route."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form route (GET to display form, POST to process prediction)."""
    if request.method == "POST":
        return handle_prediction()
    return render_template(
        "predict.html",
        seasons=ALLOWED_SEASONS,
        form_data={},
        errors=None,
    )


@app.route("/result", methods=["GET", "POST"])
def result():
    """Result route. Accepts POST directly from the form or redirects if accessed via GET."""
    if request.method == "POST":
        return handle_prediction()
    flash("Please enter agricultural parameters to generate a crop recommendation.")
    return redirect(url_for("predict"))


def handle_prediction():
    """
    Validates form data, runs Random Forest model inference,
    persists result to SQLite if user is logged in, and renders result page.
    """
    model, metadata, importances = get_model()
    if model is None:
        error_msg = "Trained model not found. Please run train_model.py first."
        return render_template(
            "predict.html",
            seasons=ALLOWED_SEASONS,
            form_data=request.form.to_dict(),
            errors=[error_msg],
        ), 503

    form_dict = request.form.to_dict()
    features_df, cleaned_inputs, errors = validate_and_preprocess_input(form_dict)

    if errors:
        return render_template(
            "predict.html",
            seasons=ALLOWED_SEASONS,
            form_data=form_dict,
            errors=errors,
        ), 400

    try:
        # Perform model prediction using features in exact order:
        # [N, P, K, temperature, humidity, rainfall]
        # (Location is strictly contextual and NEVER passed to the ML model)
        prediction_raw = model.predict(features_df)[0]
        recommended_crop = str(prediction_raw).capitalize()

        # Dynamic crop agronomic profile
        crop_profile = get_crop_profile(prediction_raw)

        # Retrieve prediction confidence/probability if available
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            confidence = round(float(max(probs)) * 100, 2)

        # Connect to Database: Store prediction if user is logged in
        saved_to_history = False
        if "user_id" in session:
            loc_val = None
            raw_loc = cleaned_inputs.get("location")
            if isinstance(raw_loc, dict):
                loc_val = raw_loc.get("display")
            elif raw_loc:
                loc_val = str(raw_loc)

            try:
                save_prediction(
                    user_id=session["user_id"],
                    N=cleaned_inputs["N"],
                    P=cleaned_inputs["P"],
                    K=cleaned_inputs["K"],
                    temperature=cleaned_inputs["temperature"],
                    humidity=cleaned_inputs["humidity"],
                    rainfall=cleaned_inputs["rainfall"],
                    season=cleaned_inputs.get("season"),
                    location=loc_val,
                    recommended_crop=recommended_crop,
                )
                saved_to_history = True
            except Exception:
                saved_to_history = False

        return render_template(
            "result.html",
            crop=recommended_crop,
            inputs=cleaned_inputs,
            location=cleaned_inputs.get("location"),
            confidence=confidence,
            model_info=metadata,
            feature_importances=importances,
            crop_profile=crop_profile,
            saved_to_history=saved_to_history,
        )
    except Exception as e:
        error_msg = "An error occurred while generating crop recommendation. Please verify your inputs and try again."
        return render_template(
            "predict.html",
            seasons=ALLOWED_SEASONS,
            form_data=form_dict,
            errors=[error_msg],
        ), 500


# -----------------------------------------------------------------------------
# User Authentication Routes (Register, Login, Logout)
# -----------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    """Farmer registration route."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if not name or not email or not password or not confirm_password:
            errors.append("All fields are required.")
        if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            errors.append("Please enter a valid email address.")
        if password and len(password) < 6:
            errors.append("Password must be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Password confirmation does not match.")

        if not errors:
            existing_user = get_user_by_email(email)
            if existing_user:
                errors.append("This email is already registered. Please log in or use a different email.")

        if errors:
            return render_template(
                "register.html",
                form_data={"name": name, "email": email},
                errors=errors,
            ), 400

        # Securely hash password before storing
        password_hash = generate_password_hash(password)
        create_user(name=name, email=email, password_hash=password_hash)

        flash("Registration successful! Please log in with your credentials.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form_data={}, errors=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Farmer login route with session-based authentication."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            return render_template(
                "login.html",
                error="Invalid email or password.",
                email=email,
            ), 400

        user = get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            return render_template(
                "login.html",
                error="Invalid email or password.",
                email=email,
            ), 401

    return render_template("login.html", error=None, email="")


@app.route("/logout")
def logout():
    """Clears user session and redirects to login."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
# Authenticated Farmer Dashboard, History, and Profile Routes
# -----------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    """Simple Farmer Dashboard."""
    user = get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    total_predictions = get_user_prediction_count(session["user_id"])
    predictions = get_predictions_by_user(session["user_id"])
    latest_prediction = predictions[0] if predictions else None

    return render_template(
        "dashboard.html",
        farmer_name=user["name"],
        total_predictions=total_predictions,
        latest_prediction=latest_prediction,
    )


@app.route("/history")
@login_required
def history():
    """My Prediction History route (scoped strictly to current logged-in user)."""
    predictions = get_predictions_by_user(session["user_id"])
    return render_template("history.html", predictions=predictions)


@app.route("/history/<int:prediction_id>")
@login_required
def prediction_detail(prediction_id):
    """View details of a stored prediction without recalculating."""
    prediction = get_prediction_by_id(prediction_id, user_id=session["user_id"])
    if not prediction:
        flash("Prediction record not found or access denied.", "danger")
        return redirect(url_for("history"))

    _, metadata, _ = get_model()
    crop_profile = get_crop_profile(prediction["recommended_crop"])
    return render_template(
        "prediction_detail.html",
        prediction=prediction,
        crop_profile=crop_profile,
        model_info=metadata,
    )


@app.route("/profile")
@login_required
def profile():
    """Simple profile page for the logged-in farmer."""
    user = get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return redirect(url_for("login"))

    total_predictions = get_user_prediction_count(session["user_id"])
    return render_template(
        "profile.html",
        user=user,
        total_predictions=total_predictions,
    )


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("predict.html", seasons=ALLOWED_SEASONS, errors=["Internal server error occurred."]), 500


if __name__ == "__main__":
    m, meta, _ = get_model()
    if m is None:
        print("\n[WARNING] Trained model not found at model/crop_model.pkl!")
        print("Please run: python train_model.py to train and save the model before serving.\n")
    else:
        print("\n[INFO] ML Model loaded successfully! Starting Flask application...\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
