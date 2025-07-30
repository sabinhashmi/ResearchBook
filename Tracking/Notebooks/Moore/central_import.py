# Core Python Libraries
import os
import warnings

# Scientific Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ROOT File Handling
import uproot

# Sklearn - Preprocessing, Models, Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

# Imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# CatBoost
from catboost import CatBoostClassifier, Pool

# SHAP for model explainability
import shap

# Suppress warnings globally
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(40)

# Instantiate common reusable objects
scaler = MinMaxScaler()
sampler = RandomUnderSampler(random_state=51)

# Classifier Models to Evaluate
models_ = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(),
    CatBoostClassifier(verbose=False)
]

modelNames = [
    "Logistic", "DecisionTree", "RandomForest", "GradientBoosting",
    "AdaBoost", "KNN", "MLP", "CatBoost"
]

def logit(x):
    """Calculate logistic function probability"""
    print("Probability Value:", 1 / (1 + np.exp(-x)))

def remove_outliers(df, columns, threshold=2):
    """Remove outliers using Z-score threshold"""
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    return df[(z_scores < threshold).all(axis=1)]


def set_plot_style():
    """Clean, Helvetica-based styling with gentle spacing after title."""
    sns.set_context("notebook", font_scale=1.15)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    plt.rcParams.update({
        'figure.figsize': (10, 5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'axes.titlepad': 12,          # ← Add vertical space after title
        'axes.labelpad': 8,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.top': False,
        'ytick.right': False,
        'axes.edgecolor': '0.2',
        'axes.linewidth': 1,
        'grid.alpha': 0.4,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
    })

__all__ = [
    # Core libraries
    'os', 'warnings', 'np', 'pd', 'plt', 'sns', 'go', 'uproot',
    # Preprocessing & modeling
    'MinMaxScaler', 'train_test_split', 'scaler', 'sampler',
    # Models
    'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
    'GradientBoostingClassifier', 'AdaBoostClassifier', 'KNeighborsClassifier',
    'MLPClassifier', 'CatBoostClassifier', 'Pool',
    # Metrics
    'confusion_matrix', 'f1_score', 'roc_auc_score', 'roc_curve', 'auc',
    # Sampling
    'RandomUnderSampler', 'SMOTE',
    # Explainability
    'shap',
    # Utility functions
    'logit', 'remove_outliers', 'set_plot_style',
    # Model collections
    'models_', 'modelNames'
]