import setuptools

with open("README.md", "r") as fh:
     long_description = fh.read()
print(setuptools.find_packages(),)
setuptools.setup(
    name="phlab",
    version="0.0.0",
    authors="Andrey Geondzhian, Keith Gilmore",
    # author_email="andrey.geondzhian@gmail.com",
    description="Phonon contirbution in RIXS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geonda/RIXS.phonons/",
    packages=[],
    install_requires=['numpy','scipy','tqdm','pathos','matplotlib'],
    python_requires='>=3.6',
)
