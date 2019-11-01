

padding =  0 -1
filter_shape = 1
stride = 1

conv2d_feature_map = 64
conv2d_1_feature_map = 256

input_size = 56
output_size = 56

print(conv2d_1_feature_map * (conv2d_feature_map * (filter_shape * filter_shape) + 1))
print(16640)

print(((input_size - filter_shape + padding + stride)/stride) * \
       ((input_size - filter_shape + padding + stride)/stride))

print(output_size * output_size)


