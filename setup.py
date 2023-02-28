from setuptools import setup, find_packages

setup(
    name='uyoloseg',
    version='0.0',
    description='SOTA Real Time Semantic Segmentation Models',
    url='https://github.com/uyolo-cn/uyolo-segmentation',
    author='uyolo1314',
    author_email='uyolo1314@gmail.com',
    classifiers=[
        "Development Status :: Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    license='Apache License 2.0',
    packages=find_packages(include=['uyoloseg'], exclude=("config", "tools", "demo", "docs")),
    zip_safe=False,
)