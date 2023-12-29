import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlops-utils",
    version="0.0.1",
    author="robert-min",
    author_email="robertmin522@gmail.com",
    description="mlops utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FLYAI4/mlops-utils",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "mlflow==2.9.2",
        "scipy==1.11.4",
        ],
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ],
    python_requires='>=3.10',
)