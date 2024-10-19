from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="text_processor",
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=requirements + ['gruut[sw]'],
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
    dependency_links=[
        "https://synesthesiam.github.io/prebuilt-apps/"
    ],
)