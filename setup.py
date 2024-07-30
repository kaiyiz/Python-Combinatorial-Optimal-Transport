from setuptools import find_packages, setup

setup(
    name='PyCoOT',
    packages=find_packages(include=['cot']),
    version='0.1.1',
    description='PyCoOT: Python Combinatorial Optimal Transport',
    author='Kaiyi Zhang',
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