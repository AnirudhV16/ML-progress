<!-- @format -->

# ML Progress

## Day 1: NumPy Fundamentals

**Date**: 18 sep  
**File**: `1Numpy.ipynb`

### What I Learned Today:

- **NumPy Basics**: Array creation, properties (shape, dtype, size), and indexing
- **Array Operations**: Slicing, boolean masking, and fancy indexing
- **Mathematical Operations**: Element wise operations, linear algebra (matrix multiplication, determinants)
- **Statistical Functions**: min, max, sum with axis operations
- **Array Manipulation**: Reshaping, stacking, copying vs references

### Key Functions Practiced:

np.array(), np.zeros(), np.ones(), np.random.randint()
np.matmul(), np.linalg.det(), np.vstack(), np.hstack()
.reshape(), .copy(), boolean operations (&, |, ~)

---

## Kaggle Learn – Intro to Machine Learning

**Folder**: `Intro to Machine Learning Kaggle`

I completed the entire [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course on Kaggle.  
Here are the lessons I worked through (with links to my notebooks):

1. How Models Work
2. [Basic Data Exploration](https://www.kaggle.com/code/anirudh4v/exercise-explore-your-data)
3. [Your First Machine Learning Model](https://www.kaggle.com/code/anirudh4v/exercise-your-first-machine-learning-model)
4. [Model Validation](https://www.kaggle.com/code/anirudh4v/exercise-model-validation)
5. [Underfitting and Overfitting](https://www.kaggle.com/code/anirudh4v/exercise-underfitting-and-overfitting)
6. [Random Forests](https://www.kaggle.com/code/anirudh4v/exercise-random-forests)
7. [Machine Learning Competitions](https://www.kaggle.com/code/anirudh4v/exercise-machine-learning-competitions)

---

## Day 2: Pandas Fundamentals

**Date**: 19 Sep  
**File**: `2Pandas.ipynb`

### What I Learned Today:

- **Pandas Basics**: Series, DataFrame creation
- **Indexing & Selection**: `.loc[]`, `.iloc[]`, conditional filtering
- **Descriptive Statistics**: `.describe()`, `.info()`, `.value_counts()`
- **Data Cleaning**: handling NaN values, renaming columns, dropping rows/columns
- **Data Manipulation**: groupby, aggregation, sorting, merging/joining DataFrames

### Key Functions Practiced:

pd.Series(), pd.DataFrame(), df.loc[], df.iloc[]  
df.describe(), df.info(), df.groupby(), df.sort_values(), df.merge()

---

## Day 3: Kaggle Learn – Intermediate Machine Learning

**Date**: 20 Sep  
**Folder**: `Intermediate ML Kaggle`

I completed the entire [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) course on Kaggle.  
Here are the lessons I worked through (with links to my notebooks):

1. [Introduction](https://www.kaggle.com/code/anirudh4v/exercise-introduction)
2. [Missing Values](https://www.kaggle.com/code/anirudh4v/exercise-missing-values)
3. [Categorical Variables](https://www.kaggle.com/code/anirudh4v/exercise-categorical-variables)
4. [Pipelines](https://www.kaggle.com/code/anirudh4v/exercise-pipelines)
5. [Cross-Validation](https://www.kaggle.com/code/anirudh4v/exercise-cross-validation)
6. [XGBoost](https://www.kaggle.com/code/anirudh4v/exercise-xgboost)
7. [Data Leakage](https://www.kaggle.com/code/anirudh4v/exercise-data-leakage)

### Key Concepts Learned:

- Handling **missing values** effectively (dropping, imputation strategies)
- Working with **categorical variables** (ordinal encoding, one-hot encoding)
- Building and managing **pipelines** for clean workflows
- Using **cross-validation** to evaluate model performance reliably
- Implementing **XGBoost**, a powerful gradient boosting algorithm
- Identifying and preventing **data leakage** in ML projects

---

## Day 4: Coursera – Supervised Machine Learning: Regression and Classification

**Date**: 21 Sep  
**File**: `4ML_course1_Week1.ipynb`

I enrolled in the **Machine Learning Specialization by Coursera (DeepLearning.AI & Stanford)** today and
completed **Week 1** of the first course: _Supervised Machine Learning: Regression and Classification_.

### Week 1: Introduction to Machine Learning

#### Topics Covered:

- **Welcome & Applications of ML**
  - What ML is and where it’s applied in the real world
- **Core ML Concepts**
  - What is machine learning?
  - Supervised learning
  - Unsupervised learning
- **Tools for ML**
  - Introduction to Jupyter Notebooks
- **Regression**
  - Linear regression model
- **Cost Function**
  - Formula and intuition
  - Visualizing the cost function
  - Visualization examples
- **Gradient Descent**
  - Concept and implementation
  - Gradient descent intuition
  - Learning rate
  - Gradient descent for linear regression

---

## Day 5: Coursera – Supervised Machine Learning: Regression and Classification

**Date**: 22 Sep  
**File**: `5ML_course1_Week2.ipynb`

I completed **Week 2** of the first course in the Machine Learning Specialization on Coursera.

### Week 2: Regression with Multiple Features

#### Topics Covered:

- **Multiple Features**
  - Linear regression with multiple input variables
- **Vectorization**
  - Efficient implementation with NumPy (Vectorization Part 1 & 2)
- **Gradient Descent for Multiple Linear Regression**
  - Extending gradient descent to handle multiple features
- **Feature Scaling**
  - Normalization and standardization
  - Checking gradient descent for convergence
  - Choosing the right learning rate
- **Feature Engineering**
  - Creating new features
  - Polynomial regression for non-linear relationships

### Applied Project: Housing Price Prediction (Kaggle Competition)

- Used the **House Prices: Advanced Regression Techniques** dataset (train & test splits from Kaggle).
- Implemented **SGDRegressor** (linear regression) with preprocessing steps such as scaling, one-hot encoding, and imputation.
- Submitted predictions to Kaggle and achieved a **public leaderboard score of ~17,000**.
- All implementation and experimentation are documented in the Colab notebook.

---

## Day 6: Coursera – Supervised Machine Learning: Regression and Classification

**Date:** 23 Sep  
**File:** `6ML_course1_Week3.ipynb`  
**Folder:** `Course_1`

completed **Week 3** of the first course in the **Machine Learning Specialization by Coursera**, and with this, **Course 1** completed.

### Week 3: Classification and Logistic Regression

#### Topics Covered:

- **Classification**
  - When and why classification is used in ML
- **Logistic Regression**
  - Implemented logistic regression with both gradient descent and scikit-learn
  - Explored the sigmoid function and decision boundaries
- **Cost Function for Logistic Regression**
  - Logistic loss and simplified cost function
  - Applied gradient descent to optimize parameters
- **Overfitting and Regularization**
  - Identified overfitting in models
  - Applied regularization techniques for linear and logistic regression
- Worked on optional labs for classification, logistic regression, cost function, gradient descent, and regularization

#### Learnings Applied:

Implemented logistic regression on the **Titanic dataset**, submitted predictions to the Kaggle competition, applied preprocessing, feature engineering, and model tuning to optimize performance, and documented all steps in the Colab notebook.

**Topics covered:** Logistic regression from scratch and using libraries, decision boundaries visualization, gradient descent optimization, regularization to prevent overfitting, classification metrics and evaluation.

---

## Day 7: Coursera – Advanced Learning Algorithms

**Date:** 24 Sep  
**File:** `78ML_course2_Week1.ipynb`

I started the **second course in the Machine Learning Specialization (Advanced Learning Algorithms)** on Coursera, **Week 1** (not yet completed, just made some progress today).

### Week 1: Neural Networks Basics

#### Topics Covered:

- **Introduction to Neural Networks**

  - Motivation: mimicking the human brain for applications in speech, images, text, and many more
  - Structure of artificial neurons: input → computation (weights, bias) → activation → output

- **Neural Network Architecture**

  - Input layer: receives raw features
  - Hidden layers: multiple neurons, each performing linear combination + activation
  - Output layer: often logistic regression for probability predictions

- **Key Concepts**
  - Hidden neurons act as feature learners, transforming raw inputs into useful activations
  - Activations from hidden layers serve as new features for the output layer
  - Forward propagation algorithm

#### Learnings Applied:

- Understood how neural networks generalize linear and logistic regression by stacking neurons in hidden layers
- Learned that activations from hidden layers allow the network to model complex, nonlinear relationships
- Worked through coding exercises to implement forward propagation and understand neuron computations

---

## Day 8: Coursera – Advanced Learning Algorithms

**Date:** 25 Sep  
**File:** `78ML_course2_Week1.ipynb`

Completed **Week 1** of the **second course in the Machine Learning Specialization (Advanced Learning Algorithms)**.

### Topics Covered

- **Introduction to TensorFlow & Keras**
  - Layers, activations, losses, and Sequential models
- **Dense Layer Basics**
  - Units, input shape, fully connected neurons
- **Linear Regression vs Single Neuron**
  - Using linear and sigmoid activations
- **Forward Propagation**
  - Automatic via `Sequential`
  - Manual layer calls
- **Data Handling**
  - NumPy 1D arrays vs TensorFlow 2D arrays (samples, features) for parallel computation
- **Practical Implementation**
  - Built a simple binary digit recognition model: `25 → 15 → 1`, using sigmoid activations
- **Model Workflow**
  - `model.compile()`, `model.fit()`, and `model.predict()`

---

### Notes:

This repository now contains my **NumPy** and **Pandas practice notebooks**, along with my completed **Kaggle ML courses**,
cousera **specialization in Machine Learning** documenting my daily ML learning journey.
