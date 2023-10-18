from setuptools import setup, find_packages

setup(
    name='budgetbuddy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'brewer2mpl',
        'fire',
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'budgetbuddy = budgetbuddy.cli.main:main',
        ],
    },
)
