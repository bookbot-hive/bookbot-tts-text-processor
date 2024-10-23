from setuptools import setup, find_packages
import os

def read_requirements():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(current_dir, 'requirements.txt'), 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

setup(
    name="text_processor",
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=read_requirements(),
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
    dynamic = ["version", "dependencies"],
    python_requires=">=3.10",
    dependency_links=[
        "https://synesthesiam.github.io/prebuilt-apps/"
    ],
)