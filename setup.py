import os
from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(
    name='PyCoOT',
    packages=find_packages(include=['cot']),
    version='0.1.3',
    description='PyCoOT: Python Combinatorial Optimal Transport',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Kaiyi Zhang',
    author_email = 'kaiyizhang0809@gmail.com',
    url='https://github.com/kaiyiz/Python-Combinatorial-Optimal-Transport/',
    license='MIT',
    package_data={
        'cot': ['optimaltransport.jar'],
    },
    install_requires=["numpy==1.24", "scipy==1.11", "torch==1.13", "jpype1==1.5.0"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite="tests",
    python_requires=">=3.9",
    )