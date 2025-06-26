#!/usr/bin/env python3
"""
Setup script for UrbanIndex v1.0.0 - Smart Buffer Urban Morphometric Parameter Calculation for WRF Modeling
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = """
    UrbanIndex v1.0.0: Smart Buffer Urban Morphometric Parameter Calculation for WRF Modeling
    
    Revolutionary Smart Buffer workflow for calculating urban morphometric parameters required by 
    the Weather Research and Forecasting (WRF) model's urban canopy schemes. This major version 
    features adaptive boundary detection, computational efficiency optimization through FRC_URB2D 
    thresholds, and unified feature processing with overlap resolution.
    
    Key Features:
    - Smart Buffer methodology with adaptive parameters
    - Easy Boundary alternative using OSMnx administrative boundaries  
    - FRC_URB2D threshold processing for computational efficiency
    - Parallel processing with multicore support
    - CSV output compatible with WRF w2w workflow
    - Comprehensive quality control and validation
    """

# Read requirements
try:
    with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Extract package name without version specifiers and comments
                package = line.split('>=')[0].split('==')[0].split('#')[0].strip()
                if package:
                    requirements.append(line)
except FileNotFoundError:
    # Fallback requirements if file not found
    requirements = [
        "geopandas>=0.13.0",
        "osmnx>=1.6.0", 
        "shapely>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0"
    ]

setup(
    name="urbanindex",
    version="3.1.0",
    author="[Your Name]",
    author_email="[your.email@domain.com]",
    description="Smart Buffer Urban Morphometric Parameter Calculation for WRF Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NicMan89/Urban-Index-for-WRF",
    py_modules=["UrbanIndex_v3"],  # Single module file
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0", 
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "interactive": [
            "folium>=0.14.0",
            "contextily>=1.3.0",
        ],
        "complete": [
            "folium>=0.14.0",
            "contextily>=1.3.0", 
            "pyproj>=3.4.0",
            "pytest>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "urbanindex=UrbanIndex_v3:main",
            "urbanindex-smart=UrbanIndex_v3:main",
        ],
    },
    keywords=[
        # Core functionality
        "urban morphology", "WRF", "weather modeling", "urban climate", 
        "building height", "urban parameters", "atmospheric modeling",
        # Data sources  
        "openstreetmap", "osm", "gis", "urban planning",
        # New v3.1.0 features
        "smart buffer", "adaptive boundary", "density analysis",
        "computational efficiency", "parallel processing",
        # Applications
        "urban heat island", "wind modeling", "urban canopy",
        "meteorology", "climate modeling", "urban design"
    ],
    project_urls={
        "Bug Reports": "https://github.com/NicMan89/Urban-Index-for-WRF/issues",
        "Source": "https://github.com/NicMan89/Urban-Index-for-WRF",
        "Documentation": "https://github.com/NicMan89/Urban-Index-for-WRF/blob/main/docs/documentation.md",
        "Changelog": "https://github.com/NicMan89/Urban-Index-for-WRF/blob/main/CHANGELOG.md",
        "Homepage": "https://github.com/NicMan89/Urban-Index-for-WRF",
    },
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,  # For better compatibility
    
    # Additional metadata for PyPI
    license="MIT",
    platforms=["any"],
    
    # Long description for what's new in v3.1.0
    long_description_content_type="text/markdown",
)