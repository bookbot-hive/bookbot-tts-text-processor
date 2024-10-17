from setuptools import setup, find_packages

setup(
    name="phoneme_emphasis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gruut",
        "transformers",
        "optimum",
        "onnxruntime",
    ],
    author="David Samuel Setiawan",
    author_email="davidsamuel.7878@gmail.com",
    description="A package for phoneme emphasis prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phoneme_emphasis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
