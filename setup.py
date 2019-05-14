import sys
import versioneer


try:
    from setuptools import setup, find_packages
except ImportError:
    raise sys.exit('Could not import setuptools.')


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages()
)
