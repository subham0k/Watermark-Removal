from setuptools import setup, find_packages

setup(
    name="WaterMark Remover Package",
    version="0.1.0",
    author="Varun Raju",
    author_email="your.varunrajus2003@gmail.com",
    description="A Simple Watermark Remover Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VarunRaju01/SDP1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",  # Add dependencies if needed
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
