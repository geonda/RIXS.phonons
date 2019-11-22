import setuptools

with open("README.md", "r") as fh:
     long_description = fh.read()
print(setuptools.find_packages(),)
setuptools.setup(
    name="phrixs", 
    version="0.0.0",
    author="Andrey Geondzhian",
    # author_email="andrey@example.com",
    description="Phonon contirbution in RIXS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=[],
    install_requires=['numpy','pyqtgraph','scipy','tqdm','pathos'],
    python_requires='>=3.6',
)
