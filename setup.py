import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelling-tool", # Replace with your own username
    version="0.0.14",
    author="Erin Balabukha",
    author_email="erin.balabukha@gmail.com",
    description="A tool for modelling covering data exploration, preprocessing, and model diagnostics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject_erinb",
    packages=setuptools.find_packages(),
#    install_requires=['pandas>=0.23', 'numpy>=1.15', 'matplotlib>=2.2.3', 'scikit-learn>=0.20.4'],
#    install_requires=['pandas', 'numpy', 'matplotlib', 'scikit-learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
