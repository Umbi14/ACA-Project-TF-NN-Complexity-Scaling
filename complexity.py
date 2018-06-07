def fc_flops(n_inputs, n_neurons):
    return n_inputs*n_neurons

def conv_flops(k, n_channels, n_filters, output_width, output_height):
    return 2*(k*k) + n_channels + n_filters + output_width * output_height
