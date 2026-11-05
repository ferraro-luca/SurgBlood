from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

sources = glob.glob("csrc/*.cu") + glob.glob("csrc/*.cpp")

setup(
    name='sam2_ext',
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=sources,
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
