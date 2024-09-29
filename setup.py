from setuptools import setup, find_packages

setup(name='PCANN',
      version='1.0.0',
      description='PCANN is an open source method for protein complex affinity prediction',
      author='OOLebedenko',
      url='https://github.com/OOLebedenko/PCANN',
      packages=find_packages(),
      install_requires=['torch', 'torch_geometric'])
