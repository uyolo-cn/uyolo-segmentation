from setuptools import setup, find_packages

setup(
    name='uyoloseg',
    version='0.0.1',
    description='SOTA Semantic Segmentation Models',
    url='https://github.com/uyolo1314/uyolo-segmentation',
    author='uyolo1314',
    author_email='uyolo1314@gmail.com',
    classifiers=[
        "Development Status :: Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license='Apache License 2.0',
    packages=find_packages(include=['uyoloseg'], exclude=("config", "tools", "demo")),
    zip_safe=False,
)