import os

# Base path = current directory where structure.py is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Folder structure definition (relative to repo root)
folders = [
    "app/pages",         # Streamlit multipage setup
    "data_pipeline",     # Data collection & preprocessing
    "models",            # Modeling scripts
    "notebooks",         # Experimentation notebooks
    "tests",             # Unit tests
    "assets",            # Static files (images, sample data)
    ".streamlit"         # Streamlit Cloud config
]

# Files to create (empty content)
files = [
    "app/main.py",
    "data_pipeline/__init__.py",
    "data_pipeline/fetch_financials.py",
    "data_pipeline/fetch_sentiment.py",
    "data_pipeline/fetch_macro.py",
    "data_pipeline/preprocess.py",
    "data_pipeline/config.py",
    "models/__init__.py",
    "models/arimax_model.py",
    "models/lstm_model.py",
    "models/tune_arimax.py",
    "models/tune_lstm.py",
    "models/evaluate.py",
    "models/interpret.py",
    "notebooks/EDA.ipynb",
    "notebooks/Feature_Engineering.ipynb",
    "notebooks/Model_Prototyping.ipynb",
    "tests/test_data_pipeline.py",
    "tests/test_models.py",
    "tests/test_app.py",
    "requirements.txt",
    "Dockerfile",
    ".streamlit/config.toml",
    "README.md",
    "LICENSE"
]

# Create folders
for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# Create empty files
for file_path in files:
    full_path = os.path.join(base_path, file_path)
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # Create empty file
    open(full_path, 'w').close()

print("✅ Project structure created successfully inside the existing repo!")
