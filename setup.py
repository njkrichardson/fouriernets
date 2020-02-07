import setuptools

setuptools.setup(name='fouriernets',
      version='0.1dev',
      description='Neural networks for density estimation on data with circular symmetries.',
      author='Nick Richardson',
      author_email='nrichardson@hmc.edu',
      url='https://github.com/njkrichardson/fouriernets',
      long_description=open('README.md').read(), 
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      python_requires=">=3.6",
      install_requires=[
        "autograd==1.3",
        "jupyterlab~=1.2.5",
        "matplotlib",
        "pandas"]
     )
