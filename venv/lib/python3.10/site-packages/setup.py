from setuptools import setup

setup(
    name='openai_secret_manager',
    version='0.1',
    description='A package to manage OpenAI API secrets.',
    packages=['openai_secret_manager'],
    install_requires=[
        'openai'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
