from setuptools import setup, find_packages

def find_version():
    import os
    with open(os.path.join("sepp", "__init__.py")) as file:
        for line in file:
            if line.startswith("__version__"):
                start = line.index('"')
                end = line[start+1:].index('"')
                return line[start+1:][:end]
            
long_description = ""

setup(
    name = 'sepp',
    packages = find_packages(include=["sepp*"]),
    version = find_version(),
    install_requires = [], # TODO
    python_requires = '>=3.5',
    description = 'Self-excited point process models of crime.',
    long_description = long_description,
    author = 'Matthew Daws',
    author_email = 'matthew.daws@gogglemail.com',
    url = '',
    license = 'MIT',
    keywords = [],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: GIS"
    ]
)
