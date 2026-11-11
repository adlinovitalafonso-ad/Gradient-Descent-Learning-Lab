# Optimizer Learning Lab

An interactive Streamlit application for teaching nonlinear optimization and machine-learning optimizers.

The app is designed for classroom use in **Mathematical Modeling for AI & Optimization** and allows students to observe how optimization algorithms move step by step toward an optimum.

## Features

### 1D Function Optimization
- Enter a custom one-variable function.
- Choose a starting point.
- Select an optimizer.
- Change the learning rate and stopping tolerance.
- View the first few optimization steps.
- Run the algorithm to convergence.
- Visualize the optimizer path on the function curve.

### 2D Function Optimization
- Define a quadratic function of two variables using coefficients.
- Observe the gradient at every iteration.
- View the optimization path on a contour plot.
- Compare the starting point and final solution.
- Explore the effect of different learning rates and starting points.

### Linear Regression Optimization
- Enter data manually or upload a CSV file.
- Choose the predictor and response variables.
- Optimize the regression parameters using:
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Adam
- View step-by-step updates of the intercept and slope.
- Observe changes in Mean Squared Error (MSE).
- Watch the regression line move toward the fitted solution.

### Student Access
Students enter:
- Email ID
- Roll Number
- Access Code

The app can record student activity for learning analytics.

### Faculty Analytics
A protected faculty dashboard can be used to review:
- Student participation
- Optimizer usage
- Learning rates tried
- Convergence activity
- Student experiment history
- Downloadable CSV analytics

## Project Structure

```text
optimizer-learning-lab/
│
├── app.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

`secrets.toml` should **not** be committed to GitHub.

## Local Installation

Open Command Prompt and move to the project folder:

```cmd
cd /d D:\MMAIO\GD
```

Activate your Python environment if required:

```cmd
D:\ortools_env\Scripts\activate
```

Install the required packages:

```cmd
pip install -r requirements.txt
```

Run the Streamlit app:

```cmd
python -m streamlit run app.py
```

The app should open automatically in your browser.

## Requirements

The application uses:

- Streamlit
- NumPy
- Pandas
- Matplotlib
- SymPy

If cloud analytics using Supabase is added, include:

- supabase

## Streamlit Secrets

Create the following file locally:

```text
.streamlit/secrets.toml
```

Example:

```toml
STUDENT_ACCESS_CODE = "MMAIO2026"
FACULTY_ACCESS_CODE = "FACULTY2026"
```

For cloud deployment, enter these values through Streamlit Community Cloud's **Secrets** settings instead of uploading the file to GitHub.

## Deployment

The recommended deployment workflow is:

```text
GitHub
   ↓
Streamlit Community Cloud
   ↓
Persistent cloud database for analytics
```

For classroom deployment with multiple students, a persistent cloud database such as Supabase/PostgreSQL is recommended instead of relying on a local SQLite database.

## Teaching Purpose

The app is intended to help students understand the relationship:

```text
Objective Function
      ↓
Gradient
      ↓
Optimizer
      ↓
Parameter Update
      ↓
Reduced Loss
      ↓
Convergence
```

It allows students to experiment with different starting points, learning rates, datasets, and optimizers to observe how the optimization path changes.

## Author

Prof. Adlino Afonso  
Goa Institute of Management
