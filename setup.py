#!/usr/bin/env python3
"""
Revolutionary O(1) Biometric Matching System
Patent Pending - Michael Derrick Jagneaux

Professional Setup Configuration
World's First Constant-Time Biometric Search Engine
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    version_file = this_directory / "src" / "__init__.py"
    if version_file.exists():
        version_content = version_file.read_text()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
        if version_match:
            return version_match.group(1)
    return "1.0.0"

# Requirements from requirements.txt
def get_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name="revolutionary-biometric-o1",
    version=get_version(),
    author="Michael Derrick Jagneaux",
    author_email="contact@revolutionarybiometrics.com",
    description="World's First O(1) Biometric Matching System - Patent Pending Technology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/revolutionarybiometrics/o1-matching-system",
    project_urls={
        "Documentation": "https://docs.revolutionarybiometrics.com",
        "Patent Filing": "https://patents.revolutionarybiometrics.com",
        "Bug Reports": "https://github.com/revolutionarybiometrics/o1-matching-system/issues",
        "Source": "https://github.com/revolutionarybiometrics/o1-matching-system",
        "Demo": "https://demo.revolutionarybiometrics.com"
    },
    
    # Package Configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "revolutionary_biometric_o1": [
            "templates/*",
            "static/*",
            "static/css/*",
            "static/js/*",
            "static/images/*",
            "config/*.yaml",
            "config/*.json",
            "tests/fixtures/*"
        ]
    },
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require={
        "development": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
            "pre-commit>=3.4.0"
        ],
        "production": [
            "gunicorn>=21.2.0",
            "uwsgi>=2.0.21",
            "supervisor>=4.2.5",
            "prometheus-client>=0.17.1"
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",
            "tensorflow-gpu>=2.13.0",
            "torch-gpu>=2.0.1"
        ],
        "advanced": [
            "tensorflow>=2.13.0",
            "torch>=2.0.1",
            "scikit-learn>=1.3.0"
        ]
    },
    
    # Python Requirements
    python_requires=">=3.9",
    
    # Entry Points for Command Line Interface
    entry_points={
        "console_scripts": [
            "o1-biometric=revolutionary_biometric_o1.cli:main",
            "o1-process=revolutionary_biometric_o1.cli:process_fingerprint",
            "o1-search=revolutionary_biometric_o1.cli:search_database",
            "o1-demo=revolutionary_biometric_o1.cli:run_demo",
            "o1-benchmark=revolutionary_biometric_o1.cli:run_benchmark",
            "o1-server=revolutionary_biometric_o1.web.app:run_server"
        ]
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Security",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Web Environment",
        "Framework :: Flask",
        "Natural Language :: English"
    ],
    
    # Keywords for Discovery
    keywords=[
        "biometric", "fingerprint", "matching", "o1", "constant-time",
        "patent", "revolutionary", "law-enforcement", "identification",
        "computer-vision", "database", "search", "performance", "scalability"
    ],
    
    # License and Legal
    license="Proprietary - Patent Pending",
    
    # Additional Metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test Configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.2",
        "pytest-cov>=4.1.0",
        "pytest-benchmark>=4.0.0"
    ],
    
    # Build Requirements
    setup_requires=[
        "wheel>=0.41.0",
        "setuptools>=68.0.0"
    ]
)