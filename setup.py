from setuptools import setup, find_packages

from kanly import __version__


def requirements(filename='requirements.in'):
    with open(filename) as f:
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        return lines


setup(
    name='kanly',
    version=__version__,
    description='Regressions! Regressions! Regressions!',
    author_email='richard.katzwer@gmail.com',
    author='Moshe Katzwer',
    packages=find_packages(),
    package_data={
        '': ['requirements.in']
    },
    install_requires=requirements(),
    data_files=[('.', ['requirements.in'])],
)

# python -m venv venv
# source venv/bin/activate
