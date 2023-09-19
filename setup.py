from setuptools import setup, find_packages

setup(
    name='RAT-SQL',
    version='1.0',
    description='A relation-aware semantic parsing model from English to SQL',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    package_data={
        '': ['*.asdl'],
    },
    install_requires=[
        'asdl~=0.1.5',
        'astor~=0.7.1',
        'attrs~=18.2.0',
        'babel~=2.7.0',
        'bpemb~=0.2.11',
        'cython~=0.29.1',
        # 'entmax~=1.0',
        'jsonnet~=0.14.0',
        'networkx~=2.2',
        'nltk~=3.4',
        'numpy~=1.21.0',
        'pyrsistent~=0.14.9',
        'pytest~=5.3.2',
        'records~=0.5.3',
        'stanford-corenlp~=3.9.2',
        'tabulate~=0.8.6',
        'torch~=1.8.0',
        'torchtext~=0.3.1',
        'tqdm',
        'transformers~=3.5.0',
        'scikit-learn',
        'protobuf~=3.20',
        'datasets'
    ],
    entry_points={"console_scripts": ["ratsql=run:main"]},
)
