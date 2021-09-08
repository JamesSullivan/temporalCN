import numpy as np
from tensorflow.keras.layers import PReLU, Conv1D,  Add, Input, Cropping2D, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow_addons.layers import SpectralNormalization

fixed_filters = 80
receptive_field_size = 127
block_size = 2

def add_temporal_block(previous, skip, kernel_size, dilation, cropping):
    """Creates a temporal block.
    Args:
        previous (tensorflow.keras.layers.Layer): previous layer to attach to on standard path.
        skip (tensorflow.keras.layers.Layer): skip layer to attach to on the skip path. Use None for intiation.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
    Returns:
        tuple of tensorflow.keras.layers.Layer: Output layers belonging to (normal path, skip path).
    """
    print(f"kernel_size: {kernel_size}  dilation: {dilation}, fixed_filters: {fixed_filters} cropping: {cropping}")
    # Identity mapping so that we hold a valid reference to previous
    block = Lambda(lambda x: x)(previous)

    for _ in range(block_size):
        convs = []
        prev_block= Lambda(lambda x: x)(block)
        convs.append(SpectralNormalization(Conv1D(fixed_filters, (kernel_size), dilation_rate=(dilation,)))(block))

        if len(convs) > 1:
            block = Concatenate(axis=1)(convs) 
        else:
            block = convs[0]
        block = BatchNormalization(axis=3, momentum=.9, epsilon=1e-4, renorm=True, renorm_momentum=.9)(block)
        block = PReLU(shared_axes=[2, 3])(block)

    # As layer output gets smaller, we need to crop less before putting output
    # on the skip path. We cannot infer this directly as tensor shapes may be variable.
    drop_left = block_size * (kernel_size - 1) * dilation
    cropping += drop_left

    if skip is None:
        previous = Conv1D(fixed_filters, 1)(previous)
    # add residual connections
    out = Add()([Cropping2D(cropping=((0,0), (drop_left, 0)))(previous), block])
    # crop from left side for skip path
    skip_out = Cropping2D(cropping=((0,0), (receptive_field_size-1-cropping, 0)))(out)
    # add current output with 1x1 conv to skip path
    if skip is not None:
        skip_out = Add()([skip, SpectralNormalization(Conv1D(fixed_filters, 1))(skip_out)])
    else:
        skip_out = SpectralNormalization(Conv1D(fixed_filters, 1))(skip_out)

    return PReLU(shared_axes=[2, 3])(out), skip_out, cropping
	
def TCN(input_dim):
    """Causal temporal convolutional network with skip connections.
       This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    Args:
        input_dim (list): Input dimension of the shape (timesteps, number of features). Timesteps may be None for variable length timeseries. 
    Returns:
        tensorflow.keras.models.Model: a non-compiled keras model.
    """ 
    # Number of dilations in order to use for the temporal blocks.
    dilations = np.array([1, 2, 4, 8, 16, 32])

    input_dim.insert(0,1)
    print(f"input_dim: {input_dim}")
    input_layer = Input(shape=input_dim)
    cropping = 0
    assert (sum(dilations) * block_size + 1) == 127, "Paper specifies receptive field size should be 127"
    
    prev_layer, skip_layer, _ = add_temporal_block(input_layer, None, 1, 1, cropping)
                
    for dilation in dilations:
        prev_layer, skip_layer, cropping = add_temporal_block(prev_layer, skip_layer, 2, dilation, cropping)

    output_layer = PReLU(shared_axes=[2, 3])(skip_layer)
    output_layer = SpectralNormalization(Conv1D(fixed_filters, kernel_size=1))(output_layer)
    output_layer = PReLU(shared_axes=[2, 3])(output_layer)
    output_layer = SpectralNormalization(Conv1D(1, kernel_size=1))(output_layer)

    return Model(input_layer, output_layer)

generator = TCN([None, 3])
discriminator = TCN([receptive_field_size, 1])