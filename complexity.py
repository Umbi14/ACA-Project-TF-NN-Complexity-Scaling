def fc_flops(n_inputs, n_neurons):
    return n_inputs*n_neurons

def conv_flops(kernel, input_shape, n_filters, output_shape):#def conv_flops(k1, k2, n_channels, n_filters, output_width, output_height):
    k1 = kernel[0]
    k2 = kernel[1]
    n_channels = input_shape.as_list()[3]
    output_width = output_shape.as_list()[1]
    output_height = output_shape.as_list()[2]
    return 2*(k1*k2) + n_channels + n_filters + output_width * output_height

# (kernel width * kernel heigth * input channles) * self.n_filters
def weights(kernel, input_shape, n_filters):
    k1 = kernel[0]
    k2 = kernel[1]
    n_channels = input_shape.as_list()[3]
    return k1 * k2 * n_channels * n_filters

# input width * input height * num of filters
def memory(output_shape, n_filters):
    input_width = output_shape.as_list()[1]
    input_height = output_shape.as_list()[2]
    return input_width * input_height * n_filters
