import setuptools

requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

with open("README.md", "r") as fh:
     long_description = fh.read()
print(setuptools.find_packages(),)
setuptools.setup(
    name="phlab",
    version="0.0.0.dev5",
    authors="Andrey Geondzhian, Keith Gilmore",
    # author_email="andrey.geondzhian@gmail.com",
    description="Phonon contirbution in RIXS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geonda/RIXS.phonons/",
    packages=setuptools.find_packages(),
    install_requires = requirements,
    python_requires='>=3.6',
)
