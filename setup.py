from setuptools import setup, find_packages

setup(
    name="sduml",
    version="0.1.0",
    description="Video Game Character State Classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pandas>=2.0.0",
            "opencv-python>=4.8.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)