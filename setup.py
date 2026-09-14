from setuptools import setup, find_packages

setup(
    name="cipo",
    version="1.0.0",
    description="Optimal portfolios of temporary climate interventions",
    packages=find_packages(include=["cipo", "cipo.*"]),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20", "scipy>=1.7"],
    extras_require={"figures": ["matplotlib>=3.3", "pandas>=1.1"]},
    license="MIT",
)
