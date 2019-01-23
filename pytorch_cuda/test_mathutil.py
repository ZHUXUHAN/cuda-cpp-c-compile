from _ext import cuda_util#找到编译好的可执行文件 直接import即可 注意cuda_util只是一个文件夹 可执行文件在这个文件夹之下
import torch
a = torch.randn(3, 5)
b = torch.randn(3, 1)
print(a)
print(b)


cuda_util.broadcast_sum(a,b,*map(int, a.size()))


# 但是使用的时候，注意，这里使用的是cuda_util文件夹名 直接.操作来调用c代码的函数，而不是调用的_cuda_util.so的可执行文件名。
# 而c代码的函数又是来自于cuda代码，所以根本上来讲，核心操作的代码实现还是在cu文件中用cuda代码来编写的。



