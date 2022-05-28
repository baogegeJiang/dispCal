import numpy as np  
# 不用numpy不加这行
from distutils.core import setup  
# 必须部分
from distutils.extension import Extension  
# 必须部分
from Cython.Distutils import build_ext  
# 必须部分

ext_modules = [Extension("cyDisp", ["cyDisp.pyx"])]#, include_dirs=[np.get_include()]),]
# filename就是你的.pyx文件前面的名字，注意后面中括号和这里的名字“最好”一致，不然有些时候会报错。
# 调用numpy就添加include_dirs参数，不然可以去掉

setup(name="cyDisp", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
# 这个name随便写，其他的格式不能变
# （更新一下，python3.8的时候这个名字要和上面的filename一样）

