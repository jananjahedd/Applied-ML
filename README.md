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
├── data/                    # Raw PSG data (not tracked by Git)
├── src/                     # Source code
│   ├── __init__.py
│   ├── main.py              # Main script for running the pipeline
│   ├── features/            # Feature extraction and preprocessing
│   ├── models/              # SVM model training and evaluation
│   └── utils/               # Utility functions (e.g., data loading)
├── notebooks/               # Jupyter notebooks for experimentation
├── tests/                   # Unit tests
│   ├── features/
│   ├── models/
│   └── test_main.py
├── README.md
├── environment.yml          # Dependency management
└── .pre-commit-config.yaml  # Pre-commit hooks configuration
```

## Data
The project uses the [Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/). Place the dataset files in the `data/` directory. The data includes:
- Electroencephalograms (EEG)
- Electrooculograms (EOG)
- Electromyograms (EMG)
- Repiration and body temperatures (only some)
- Event markers

**Note**: The `data/` directory is not tracked by Git. Ensure you download the dataset manually.

## Usage
1. **Activate the Anaconda environment**:
   ```bash
   conda activate AML
   ```

2. **Run the main script**:
   ```bash
   python src/main.py
   ```

3. **Experiment in Jupyter notebooks**:
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
