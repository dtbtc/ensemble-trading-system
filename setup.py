from setuptools import setup, find_packages

setup(
    name="blending_ensemble_trading",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'lightgbm>=3.3.2',
        'xgboost>=1.5.0',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.1',
        'PyYAML>=5.4.1',
        'joblib>=1.0.1',
    ],
    author="LI ZHIYUE",
    author_email="your.email@example.com",
    description="Research System for Cryptocurrency Trading Strategies Based on Ensemble Learning",
    python_requires='>=3.8',
) 