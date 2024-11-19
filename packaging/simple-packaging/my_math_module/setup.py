from setuptools import setup, find_packages

setup(
    name="my_math_module",                  # Package name
    version="0.1.0",                        # Package version
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple math module",
    long_description=open("README.md").read(),  # Long description from README
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_math_module",
    packages=find_packages(),               # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",                # Minimum Python version requirement
)
