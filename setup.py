from setuptools import setup, find_packages

setup(
    name="bookbot_tts_text_processor",
    packages=find_packages(),
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
    description="A package processing input text for Bookbot Optispeech TTS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bookbot-hive/bookbot-tts-text-processor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)