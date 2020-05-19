import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="moving-mnist-for-segmentation-Roshan-Rane", # Replace with your own username
    version="0.0.1",
    author="Roshan Rane",
    author_email="rosh1992@gmail.com",
    description="A package to quickly generate moving-MNIST videos for segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RoshanRane/segmentation-moving-MNIST",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)