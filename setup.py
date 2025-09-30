from setuptools import setup, find_packages

setup(
    name="ml-from-scratch",
    version="0.1",
    description="Implémentations pédagogiques d'algorithmes de Machine Learning",
    author="ZENAGUI Mohamed Elakef",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ],
)
