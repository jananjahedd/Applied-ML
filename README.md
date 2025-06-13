# Automated Sleep Stage Detection

## Project Overview
This project aims to develop a classical machine learning model for automated sleep stage classification. We want to look at how intermediate fusion of electroencephalography (EEG), electrooculogram (EOG), electromyography (EMG) and respiration signals affects the performance of classical machine learning models in sleep stage classification compared to using EEG signals alone.

## Installation

### Prerequisites
- Python 3.10
- Git
- Anaconda

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/jananjahedd/Applied-ML
   ```

2. **Set up the Anaconda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate AML
   ```

2. **Set up the virtual environment**:
   ```bash
   pip install pipenv
   pipenv install
   pipenv shell
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Project Structure
```
.
|
|— data/                    # Raw PSG data (not tracked by Git)
|— /example-data            # Data used for the application
|— /notebooks               # Jupyter notebooks for experimentation
|— /results                 # The results of the models
|— /tests                   # Unit tests
|
|-- /backend
|   |-- main.py             # Main sript for running the FASTAPI app
|   |-- src/
|   |   ├── __init__.py
|   │   ├── data/           # Files for processing the data
|   |   ├── endpoints/      # API endpoints
|   │   ├── features/       # Feature extraction file
|   │   ├── models/         # SVM model training and evaluation
|   │   └── utils/          # Utility functions (e.g., paths)
|   |   |-- schemas.py      # Schemas for the API
|   |   |-- script.py       # Runs the complete pipeline (with training)
|   |
|   |-- requirements.txt    # Requirements for the backend
|   |-- logs/               # Log files
|   |-- Dockerfile          # Docker for the backend
|
|-- /frontend
|   |-- pages/              # Scripts for the streamlit app
|   |—- app.py              # The streamlit app
|   |-- Dockerfile          # Docker file for frontend
|   |—- requirements.txt    # Requirements for the frontend
| 
|-- .dockerignore.
|-- .gitignore
|-- docker-compose.yml      # File for composing both backend and frontend
|
|—- .pre-commit-config.yaml # Pre-commit hooks configuration
|—- environment.yml         # Dependency management
|—- README.md
```

## Data
The project uses the [Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/). Place the dataset files in the `data/` directory. The data includes:
- Electroencephalograms (EEG)
- Electrooculograms (EOG)
- Electromyograms (EMG)
- Repiration and body temperatures (only some)
- Event markers

**Note**: The `data/` directory is not tracked by Git. Ensure you download the dataset manually.
The user has to ensure that they have `awscli` installed on their computer:

## Usage
1. **Activate the Anaconda environment**:
   ```bash
   conda activate AML
   ```

2. **Download the data**:
   If not already done, the user must download the data manually:
   ```bash
   pip install awscli

   aws --version

   cd Applied-ML

   aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ ./data
   ```
   If using MacOS, the user can also make use of the `caffeinate` built-in function to make the process od downloading faster:
   ```bash
   caffeinate -i aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ ./data
   ```

3. **Run the main script**:
   This is used to run the entire pipeline process which includes the preprocessing of the data, splitting (with feature engineering and data augmentation) and training of the SVM models:
   ```bash
   cd Applied-ML/backend

   python -m src.script.py
   ```

4. **Experiment in Jupyter notebooks**:
   Place exploratory code in the `notebooks/` directory

## Pre-Commit Hooks
Pre-commit hooks automatically run checks before each Git commit to ensure code quality. They enforce Code formatting with Black, PEP8 compliance with Flake8, import sorting with isort, etc.

### How to Use Pre-Commit
- After installing (`pre-commit install`), hooks run automatically on `git commit`.
- To manually run hooks on all files:
  ```bash
  pre-commit run --all-files
  ```
- To skip hooks for a commit:
  ```bash
  git commit --no-verify -m "message"
  ```

## Development Guidelines
- **Code Quality**: Use Black for formatting, Flake8 for PEP8 compliance, and isort for import sorting. Pre-commit hooks enforce these standards.
- **Version Control**: Follow Git best practices:
  - Create feature branches (`git checkout -b feature-name`)
  - Use atomic commits with meaningful messages
  - Submit pull requests for code review
- **Testing**: Add unit tests in the `tests/` directory using pytest.
- **Notebooks**: Use `notebooks/` for experimentation, but move stable code to `src/`.

## How to run the FastAPI app

After activating the Anaconda environment, the user has to manually open the app:
```bash
cd Applied-ML/backend

uvicorn main:app --reload
```
The app will start, and the user can copy the URL address and paste it on their browser with the **/docs** added at the end (e.g., `http://127.0.0.1:8000/docs`).

To run tests on the API endpoints:
```bash
pytest tests/test_main.py
pytest tests/test_pipeline.py
pytest tests/test_schemas.py
```

## Compose the Docker

To run the application locally, the user must open the FastAPI app, then the streamlit app. For this, we containerised both the backend and frontend (Docker files) to have a smoother approach for deployment:
```bash
cd Applied-ML

docker-compose up --build
```
The user can copy-paste the URL after the application has started directly on their preferred Browser. The application is intuitive and explains the necessary steps for usage. Additionally, to stop the application, the user can type `Ctrl+C` and to remove the dockers:
```bash
docker-compose down
```
