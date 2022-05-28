#!/usr/bin/env python 
# 不用numpy不加这行
from distutils.core import setup  
# 必须部分
from distutils.extension import Extension  
# 必须部分
from Cython.Distutils import build_ext 


setup(
    name='dispCal',
    version='0.1.4',
    description='Calculate Surface Wave Dispersion',
    author='baogege(Jiang Yiran)',
    author_email="baogege@pku.edu.cn",
    url="https://github.com/baogegeJiang/dispCal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={
        'dispCal': 'src'
    },
    packages=[
        'dispCal',
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            name='cyDisp',
            sources=['src/cyDisp.pyx'], # noqa
            )
        ]
)
