from setuptools import setup, find_packages

setup(
    name='budgetbuddy',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fire',
        'matplotlib',
        'numpy',
        'openpyxl',
        'pandas',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'budgetbuddy = budgetbuddy.cli.main:main',
        ],
    },
)
