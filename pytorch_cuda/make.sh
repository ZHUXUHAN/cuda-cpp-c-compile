cd src/
nvcc -c -o mathutil_cuda_kernel.cu.o mathutil_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ..
python3 build.py