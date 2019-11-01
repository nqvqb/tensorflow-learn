
import os
import time
import numpy as np
import tensorflow as tf

# https://blog.csdn.net/linuxwindowsios/article/details/67635867

import numpy as np
t = np.random.randint(low=0,high=5,size=10)
print('t', t)

# calculate the norm of the random aray
# 计算L2范数
l2norm4t = np.linalg.norm(t)
print('l2norm4t', l2norm4t)

# clip norm
# 随机数规约
clip_norm = 5
transformed_t = t *clip_norm/l2norm4t
print('transformed_t',transformed_t)

# validation
print('the norm should be restricted within clip_norm', clip_norm)
print('validate the norm after clip norm', np.linalg.norm(transformed_t))