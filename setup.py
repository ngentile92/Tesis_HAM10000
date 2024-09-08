from setuptools import setup, find_packages

# Leer las dependencias desde requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Tesis_HAM10000", 
    version="0.1",
    packages=find_packages(),  
    install_requires=requirements,  
    include_package_data=True,
    description="Project for handling HAM10000 dataset",
    author="Nicolas Gentile",
)
