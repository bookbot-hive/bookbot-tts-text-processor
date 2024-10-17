from setuptools import setup, find_packages

setup(
    name="phoneme_emphasis",
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        "torch",
        "gruut==2.4.0",
        "transformers==4.45.2",
        "optimum[onnxruntime]",
    ],
    author="David Samuel Setiawan",
    author_email="davidsamuel.7878@gmail.com",
    description="A package for phoneme emphasis prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bookbot-hive/phoneme_emphasis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)