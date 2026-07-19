import re
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def get_version():
    init_text = (ROOT / "kanly" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        init_text,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Unable to find __version__ in kanly/__init__.py")
    return match.group(1)


def requirements(filename="requirements.in"):
    with (ROOT / filename).open(encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        ]


setup(
    name="kanly",
    version=get_version(),
    description="Regressions! Regressions! Regressions!",
    author_email="richard.katzwer@gmail.com",
    author="Moshe Katzwer",
    packages=find_packages(),
    package_data={"": ["requirements.in"]},
    install_requires=requirements(),
    data_files=[(".", ["requirements.in"])],
)


# python -m venv venv
# source venv/bin/activate