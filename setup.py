from setuptools import setup, find_packages

setup(
    name='srf-attention',
    version='1.0.0',
    packages=find_packages(),
    author='Not a Russian Teenager!',
    author_email='notarussianteenager@gmail.com',
    description='Simplex random feature attention in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/notarussianteenager/srf-attention',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'torch',
        'scipy',
        'einops'
    ],
    include_package_data=True,
    python_requires='>=3.6',
)


