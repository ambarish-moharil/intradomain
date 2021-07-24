from setuptools import setup

setup(
    name='intradomain_toolkit',
    version='0.1.0',    
    description='A Context Disambiguation Package for Natutal Language Documnents (English)',
    url='https://github.com/ambarish-moharil/intradomain_toolkit',
    author='Ambarish Moharil',
    author_email='ambarish.m02@gmail.com',
    license='MIT License',
    packages=['intradomain_toolkit'],
    install_requires=['pandas',
                      'numpy',                     
                      'transformers',
                      'matplotlib',
                      'nltk',
                      'torch'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
