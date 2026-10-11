import os
import sqlite3
import unittest
from werkzeug.security import check_password_hash
from app import app, get_model
from utils.database import DB_PATH, get_db_connection


class SmartCropAITestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    # =========================================================================
    # ORIGINAL EXISTING TESTS (1 - 10)
    # =========================================================================

    def test_01_model_loaded(self):
        """Verify the trained ML model and metadata load properly."""
        model, metadata, importances = get_model()
        self.assertIsNotNone(model, "Model should not be None")
        self.assertIsNotNone(metadata, "Metadata should not be None")
        self.assertIn("train_accuracy", metadata)
        self.assertIn("test_accuracy", metadata)
        self.assertIsNotNone(importances, "Feature importances should not be None")
        self.assertEqual(len(importances), 6, "Should have 6 feature importances")
        print(f"\n[TEST PASS] Model loaded successfully. Test Accuracy: {metadata['test_accuracy'] * 100:.2f}%")

    def test_02_home_page(self):
        """Test GET / renders Home Page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SmartCrop AI", response.data)
        self.assertIn(b"AI-Based Smart Crop Recommendation System", response.data)
        self.assertIn(b"Start Prediction", response.data)
        print("[TEST PASS] Home page loads correctly.")

    def test_03_predict_page_get(self):
        """Test GET /predict renders form."""
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Crop Recommendation Form", response.data)
        self.assertIn(b"Nitrogen (N)", response.data)
        self.assertIn(b"Phosphorus (P)", response.data)
        self.assertIn(b"Potassium (K)", response.data)
        self.assertIn(b"Temperature", response.data)
        self.assertIn(b"Humidity", response.data)
        self.assertIn(b"Rainfall", response.data)
        self.assertIn(b"Season", response.data)
        print("[TEST PASS] Prediction form loads with all required fields.")

    def test_04_valid_prediction_rice(self):
        """Test valid agricultural input for Rice-like conditions."""
        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.87",
            "humidity": "82.00",
            "rainfall": "202.93",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended Crop", response.data)
        self.assertIn(b"Rice", response.data)
        self.assertIn(b"Input Summary", response.data)
        self.assertIn(b"90.0 mg/kg", response.data)
        self.assertIn(b"Kharif", response.data)
        self.assertIn(b"Try Again", response.data)
        print("[TEST PASS] Valid Rice prediction successful.")

    def test_05_valid_prediction_coffee(self):
        """Test valid agricultural input for Coffee-like conditions."""
        payload = {
            "N": "104",
            "P": "18",
            "K": "30",
            "temperature": "26.5",
            "humidity": "58.0",
            "rainfall": "160.0",
            "season": "Winter",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended Crop", response.data)
        self.assertIn(b"Coffee", response.data)
        self.assertIn(b"Winter", response.data)
        print("[TEST PASS] Valid Coffee prediction successful.")

    def test_06_missing_values(self):
        """Test missing required fields returns 400 with helpful error messages."""
        payload = {
            "N": "",
            "P": "40",
            "K": "40",
            "temperature": "25.0",
            "humidity": "70.0",
            "rainfall": "100.0",
            "season": "",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Nitrogen (N) is required", response.data)
        self.assertIn(b"Please select a season", response.data)
        print("[TEST PASS] Missing values handled with validation errors.")

    def test_07_negative_values(self):
        """Test negative values returns 400 with helpful error message."""
        payload = {
            "N": "-10",
            "P": "40",
            "K": "40",
            "temperature": "25.0",
            "humidity": "70.0",
            "rainfall": "-50.0",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"cannot be less than 0.0", response.data)
        print("[TEST PASS] Negative values rejected with validation errors.")

    def test_08_invalid_humidity(self):
        """Test humidity > 100 returns 400."""
        payload = {
            "N": "50",
            "P": "40",
            "K": "40",
            "temperature": "25.0",
            "humidity": "150.0",
            "rainfall": "100.0",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Humidity cannot exceed 100.0", response.data)
        print("[TEST PASS] Invalid humidity (>100) rejected.")

    def test_09_non_numeric_values(self):
        """Test non-numeric characters return 400."""
        payload = {
            "N": "abc",
            "P": "40",
            "K": "40",
            "temperature": "warm",
            "humidity": "high",
            "rainfall": "heavy",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"must be a valid number", response.data)
        print("[TEST PASS] Non-numeric input rejected cleanly.")

    def test_10_get_result_redirects(self):
        """Test direct GET /result redirects to predict form."""
        response = self.client.get("/result")
        self.assertEqual(response.status_code, 302)
        print("[TEST PASS] Direct GET /result redirects cleanly.")

    # =========================================================================
    # NEW DATABASE, AUTHENTICATION, AND PREDICTION HISTORY TESTS (11 - 25)
    # =========================================================================

    def test_11_database_and_tables_exist(self):
        """1-4: Verify SQLite database and required tables exist with exact schemas."""
        self.assertTrue(os.path.exists(DB_PATH), f"Database file should exist at {DB_PATH}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check users table
        users_cols = [r["name"] for r in cursor.execute("PRAGMA table_info(users)").fetchall()]
        for col in ["id", "name", "email", "password_hash", "created_at"]:
            self.assertIn(col, users_cols, f"users table missing {col}")

        # Check predictions table
        preds_cols = [r["name"] for r in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
        for col in ["id", "user_id", "created_at", "N", "P", "K", "temperature", "humidity", "rainfall", "season", "location", "recommended_crop"]:
            self.assertIn(col, preds_cols, f"predictions table missing {col}")
        conn.close()
        print("[TEST PASS] Database and tables (users, predictions) created automatically with correct schemas.")

    def test_12_registration_workflow(self):
        """5, 7, 8: Test farmer registration with password confirmation and hashing."""
        unique_email = "farmer_test12@example.com"
        reg_payload = {
            "name": "Kisan Bhai",
            "email": unique_email,
            "password": "SecurePassword123",
            "confirm_password": "SecurePassword123",
        }
        response = self.client.post("/register", data=reg_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Registration successful", response.data)
        self.assertIn(b"Farmer Login", response.data)

        # Verify password is saved as a secure hash in SQLite, NOT plain-text
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE email = ?", (unique_email,))
        row = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertNotEqual(row["password_hash"], "SecurePassword123", "Password MUST NOT be stored in plain-text")
        self.assertTrue(check_password_hash(row["password_hash"], "SecurePassword123"))
        print("[TEST PASS] Registration creates account and stores securely hashed password.")

    def test_13_duplicate_email_rejected(self):
        """6: Test that duplicate email registration is rejected."""
        email = "duplicate_test@example.com"
        payload = {
            "name": "First Farmer",
            "email": email,
            "password": "Password123",
            "confirm_password": "Password123",
        }
        self.client.post("/register", data=payload)

        # Attempt to register again with same email
        dup_payload = {
            "name": "Second Farmer",
            "email": email,
            "password": "Password456",
            "confirm_password": "Password456",
        }
        response = self.client.post("/register", data=dup_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"already registered", response.data)
        print("[TEST PASS] Duplicate email registration is rejected with 400 error.")

    def test_14_password_confirmation_mismatch(self):
        """7: Test that mismatched passwords reject registration."""
        payload = {
            "name": "Mismatch Farmer",
            "email": "mismatch@example.com",
            "password": "Password123",
            "confirm_password": "DifferentPassword456",
        }
        response = self.client.post("/register", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Password confirmation does not match", response.data)
        print("[TEST PASS] Mismatched password confirmation rejected.")

    def test_15_login_success(self):
        """9: Test login with valid credentials creates session and redirects to dashboard."""
        email = "login_test@example.com"
        reg_payload = {
            "name": "Anil Kumar",
            "email": email,
            "password": "FarmerPassword1",
            "confirm_password": "FarmerPassword1",
        }
        self.client.post("/register", data=reg_payload)

        login_payload = {
            "email": email,
            "password": "FarmerPassword1",
        }
        response = self.client.post("/login", data=login_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome, Anil Kumar", response.data)
        self.assertIn(b"Farmer Dashboard", response.data)
        with self.client.session_transaction() as sess:
            self.assertIn("user_id", sess)
            self.assertEqual(sess["user_name"], "Anil Kumar")
        print("[TEST PASS] Login with valid credentials establishes session and displays dashboard.")

    def test_16_invalid_login_rejected(self):
        """10: Test invalid login displays exact requirement 'Invalid email or password.'."""
        # Non-existent user
        response = self.client.post("/login", data={"email": "nonexistent@example.com", "password": "wrong"})
        self.assertEqual(response.status_code, 401)
        self.assertIn(b"Invalid email or password.", response.data)

        # Existing user with wrong password
        email = "wrongpw_test@example.com"
        self.client.post("/register", data={
            "name": "Tester", "email": email, "password": "CorrectPassword", "confirm_password": "CorrectPassword"
        })
        response2 = self.client.post("/login", data={"email": email, "password": "WrongPassword"})
        self.assertEqual(response2.status_code, 401)
        self.assertIn(b"Invalid email or password.", response2.data)
        print("[TEST PASS] Invalid login rejected with exact error message.")

    def test_17_logout_workflow(self):
        """11: Test logout clears session and redirects."""
        email = "logout_test@example.com"
        self.client.post("/register", data={
            "name": "Logout Farmer", "email": email, "password": "Pass123456", "confirm_password": "Pass123456"
        })
        self.client.post("/login", data={"email": email, "password": "Pass123456"})

        # Now logout
        response = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"logged out successfully", response.data)
        with self.client.session_transaction() as sess:
            self.assertNotIn("user_id", sess)
        print("[TEST PASS] Logout successfully clears session.")

    def test_18_dashboard_cards_and_protection(self):
        """12: Test dashboard displays 4 required cards and enforces authentication."""
        # Unauthenticated access redirects to login
        guest_resp = self.client.get("/dashboard")
        self.assertEqual(guest_resp.status_code, 302)
        self.assertIn("/login", guest_resp.headers["Location"])

        # Authenticated access
        email = "dash_test@example.com"
        self.client.post("/register", data={
            "name": "Sunita Devi", "email": email, "password": "Pass123456", "confirm_password": "Pass123456"
        })
        self.client.post("/login", data={"email": email, "password": "Pass123456"})

        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome, Sunita Devi", response.data)
        self.assertIn(b"New Prediction", response.data)
        self.assertIn(b"Prediction History", response.data)
        self.assertIn(b"My Profile", response.data)
        self.assertIn(b"Logout", response.data)
        print("[TEST PASS] Dashboard renders all 4 required cards and requires login.")

    def test_19_guest_prediction_not_saved(self):
        """13: Existing crop prediction continues to work for guest users without database save."""
        # Ensure client has no session
        self.client.get("/logout")
        conn = get_db_connection()
        count_before = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()

        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.87",
            "humidity": "82.00",
            "rainfall": "202.93",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rice", response.data)
        self.assertIn(b"Guest Mode", response.data)

        conn = get_db_connection()
        count_after = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        conn.close()
        self.assertEqual(count_before, count_after, "Guest predictions should not be stored in database")
        print("[TEST PASS] Guest prediction works without saving to database.")

    def test_20_authenticated_prediction_saved_with_location(self):
        """14, 15, 20: Test logged-in prediction is automatically stored in SQLite with location."""
        email = "pred_saver@example.com"
        self.client.post("/register", data={
            "name": "Saver Farmer", "email": email, "password": "Pass123456", "confirm_password": "Pass123456"
        })
        self.client.post("/login", data={"email": email, "password": "Pass123456"})

        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.87",
            "humidity": "82.00",
            "rainfall": "202.93",
            "season": "Kharif",
            "location_mode": "manual",
            "village": "Baramati",
            "district": "Pune",
            "state": "Maharashtra",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rice", response.data)
        self.assertIn(b"Saved to History", response.data)

        # Verify record in SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
        pred = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(pred)
        self.assertEqual(pred["recommended_crop"], "Rice")
        self.assertEqual(pred["season"], "Kharif")
        self.assertIn("Baramati", pred["location"])
        self.assertIn("Pune", pred["location"])
        self.assertEqual(float(pred["N"]), 90.0)
        print("[TEST PASS] Logged-in prediction is saved with location and agricultural features.")

    def test_21_prediction_without_location(self):
        """21: Test prediction works when location is not provided."""
        payload = {
            "N": "104",
            "P": "18",
            "K": "30",
            "temperature": "26.5",
            "humidity": "58.0",
            "rainfall": "160.0",
            "season": "Winter",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Coffee", response.data)
        print("[TEST PASS] Prediction works completely without location.")

    def test_22_prediction_history_display(self):
        """16: Test prediction history page displays user predictions in order."""
        email = "history_viewer@example.com"
        self.client.post("/register", data={
            "name": "History Farmer", "email": email, "password": "Pass123456", "confirm_password": "Pass123456"
        })
        self.client.post("/login", data={"email": email, "password": "Pass123456"})

        # Submit a prediction
        self.client.post("/predict", data={
            "N": "90", "P": "42", "K": "43", "temperature": "20.87", "humidity": "82.00", "rainfall": "202.93", "season": "Kharif"
        })

        # View history
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"My Prediction History", response.data)
        self.assertIn(b"Rice", response.data)
        self.assertIn(b"Kharif", response.data)
        self.assertIn(b"View Details", response.data)
        print("[TEST PASS] Prediction history displays logged-in user records correctly.")

    def test_23_user_isolation(self):
        """17: Test User A cannot see User B's predictions."""
        # Create User A and make a prediction
        email_a = "user_a@example.com"
        self.client.post("/register", data={"name": "Farmer A", "email": email_a, "password": "Pass123456", "confirm_password": "Pass123456"})
        self.client.post("/login", data={"email": email_a, "password": "Pass123456"})
        self.client.post("/predict", data={"N": "90", "P": "42", "K": "43", "temperature": "20.87", "humidity": "82.00", "rainfall": "202.93", "season": "Kharif"})

        # Get User A's prediction id
        conn = get_db_connection()
        pred_a = conn.execute("SELECT id FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        pred_a_id = pred_a["id"]

        # Logout User A, register & login User B
        self.client.get("/logout")
        email_b = "user_b@example.com"
        self.client.post("/register", data={"name": "Farmer B", "email": email_b, "password": "Pass123456", "confirm_password": "Pass123456"})
        self.client.post("/login", data={"email": email_b, "password": "Pass123456"})

        # User B visits /history: should NOT see User A's prediction
        history_resp = self.client.get("/history")
        self.assertEqual(history_resp.status_code, 200)
        self.assertIn(b"No Saved Predictions Yet", history_resp.data)

        # User B attempts to access User A's prediction by URL
        detail_resp = self.client.get(f"/history/{pred_a_id}", follow_redirects=True)
        self.assertEqual(detail_resp.status_code, 200)
        self.assertIn(b"access denied", detail_resp.data)
        print("[TEST PASS] User isolation verified: User A's predictions are completely hidden from User B.")

    def test_24_prediction_details_stored_display(self):
        """18: Test prediction details page displays stored values without recalculation."""
        email = "detail_test@example.com"
        self.client.post("/register", data={"name": "Detail Farmer", "email": email, "password": "Pass123456", "confirm_password": "Pass123456"})
        self.client.post("/login", data={"email": email, "password": "Pass123456"})
        self.client.post("/predict", data={
            "N": "104", "P": "18", "K": "30", "temperature": "26.5", "humidity": "58.0", "rainfall": "160.0", "season": "Winter"
        })

        conn = get_db_connection()
        pred = conn.execute("SELECT id FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        response = self.client.get(f"/history/{pred['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Coffee", response.data)
        self.assertIn(b"104", response.data)
        self.assertIn(b"Winter", response.data)
        self.assertIn(b"Back to History", response.data)
        self.assertIn(b"Print / Save Record", response.data)
        print("[TEST PASS] Prediction details displays stored values accurately.")

    def test_25_profile_page(self):
        """19: Test profile page displays farmer name, email, and hides password hash."""
        email = "profile_test@example.com"
        self.client.post("/register", data={"name": "Gita Devi", "email": email, "password": "SecretPassword123", "confirm_password": "SecretPassword123"})
        self.client.post("/login", data={"email": email, "password": "SecretPassword123"})

        response = self.client.get("/profile")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Gita Devi", response.data)
        self.assertIn(b"profile_test@example.com", response.data)
        self.assertIn(b"Member Since", response.data)
        # Verify no password hash is exposed
        self.assertNotIn(b"pbkdf2", response.data)
        self.assertNotIn(b"scrypt", response.data)
        self.assertNotIn(b"SecretPassword123", response.data)
        print("[TEST PASS] Profile page displays user metadata securely without exposing password hashes.")

    def test_26_model_insights_and_print_elements(self):
        """23, 24, 25: Test Model Insights, feature importances, and print button are present."""
        payload = {
            "N": "90",
            "P": "42",
            "K": "43",
            "temperature": "20.87",
            "humidity": "82.00",
            "rainfall": "202.93",
            "season": "Kharif",
        }
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Machine Learning Model Insights", response.data)
        self.assertIn(b"Feature Importance", response.data)
        self.assertIn(b"Print / Save Result", response.data)
        self.assertIn(b"Random Forest Classifier", response.data)
        print("[TEST PASS] Existing Model Insights, feature importances, and Print elements verified.")


if __name__ == "__main__":
    unittest.main()
