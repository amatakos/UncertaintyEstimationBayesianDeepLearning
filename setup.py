from setuptools import setup, find_packages

setup(
    name='src',
    version='0.0.1',
    url='https://github.com/AaltoML/InternshipMatakos',
    author=['Alexandros Matakos'],
    author_email='',
    description='Summer internship on approximate inference in BNNs with pytorch.',
    packages=find_packages(),
    install_requires=[
        'torch == 1.11.0',
        'pyro-ppl == 1.4.0'
    ],
)
