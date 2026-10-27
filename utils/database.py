import os
import sqlite3

# Resolve absolute path to database/smartcrop.db (can be overridden by SQLITE_DB_PATH for persistent disks)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DEFAULT_DB_PATH = os.path.join(DB_DIR, "smartcrop.db")
DB_PATH = os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)


def get_db_connection(db_path=None):
    """
    Establishes and returns a connection to the SQLite database.
    Ensures the parent database directory exists, enforces foreign keys,
    and configures sqlite3.Row for dict-like column access.
    """
    target_path = db_path or DB_PATH
    db_folder = os.path.dirname(target_path)
    if db_folder and not os.path.exists(db_folder):
        os.makedirs(db_folder, exist_ok=True)

    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=None):
    """
    Initializes the SQLite database when Flask starts:
    1. Checks/creates the database directory if needed.
    2. Creates the 'users' table if it doesn't exist.
    3. Creates the 'predictions' table if it doesn't exist.
    Preserves all existing data without resetting or dropping tables.
    """
    target_path = db_path or DB_PATH
    db_folder = os.path.dirname(target_path)
    if db_folder and not os.path.exists(db_folder):
        os.makedirs(db_folder, exist_ok=True)

    conn = get_db_connection(target_path)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create predictions table with foreign key to users.id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            N REAL NOT NULL,
            P REAL NOT NULL,
            K REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            rainfall REAL NOT NULL,
            season TEXT,
            location TEXT,
            recommended_crop TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


def create_user(name, email, password_hash, db_path=None):
    """
    Inserts a new user record into the database.
    Email is normalized to lowercase.
    Returns the newly created user ID.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
        (name.strip(), email.strip().lower(), password_hash),
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_user_by_email(email, db_path=None):
    """
    Fetches a user record by email address (case-insensitive).
    Returns sqlite3.Row or None if not found.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE LOWER(email) = LOWER(?)",
        (email.strip(),),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id, db_path=None):
    """
    Fetches a user record by its primary key ID.
    Returns sqlite3.Row or None if not found.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def save_prediction(
    user_id,
    N,
    P,
    K,
    temperature,
    humidity,
    rainfall,
    season,
    location,
    recommended_crop,
    db_path=None,
):
    """
    Saves a crop prediction record for a logged-in user.
    Returns the newly created prediction ID.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            user_id, N, P, K, temperature, humidity, rainfall, season, location, recommended_crop
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            float(N),
            float(P),
            float(K),
            float(temperature),
            float(humidity),
            float(rainfall),
            season or None,
            location or None,
            recommended_crop,
        ),
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id


def get_predictions_by_user(user_id, db_path=None):
    """
    Retrieves all predictions belonging to a specific user,
    ordered by newest prediction first.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC, id DESC
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_prediction_by_id(prediction_id, user_id=None, db_path=None):
    """
    Retrieves a specific prediction by ID.
    If user_id is provided, ensures that only the owning user can retrieve it.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id),
        )
    else:
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return row


def get_user_prediction_count(user_id, db_path=None):
    """
    Returns the total number of predictions saved by a specific user.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?",
        (user_id,),
    )
    result = cursor.fetchone()
    conn.close()
    return result["total"] if result else 0
