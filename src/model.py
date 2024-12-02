from keras import layers, optimizers, losses, metrics, Model


def convolutional_block(input, num_filters):
    '''
    Computes double data convolution for the U-Net model.

    Parameters:
        input: Tensor output from the previous layer.
        num_filters: Number of filters or 'width' for the layer.

    Returns:
        Processed input data.
    '''
    output = layers.Conv2D(num_filters, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal')(input)
    output = layers.Conv2D(num_filters, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal')(output)
    return output


def encoder_block(input, num_filters):
    '''
    Handles data downsampling for the U-Net model.

    Parameters:
        input: Tensor output from the previous encoder layer.
        num_filters: Number of filters or 'width' for the layer.

    Returns:
        tuple:
            - Processed input data from the convolutional block.
            - Output data after pooling (downsampling).
    '''
    output = convolutional_block(input, num_filters)
    pooled = layers.MaxPooling2D(2)(output)
    return output, pooled


def decoder_block(input, skip_connection, num_filters):
    '''
    Handles data upsampling for the U-Net model.

    Parameters:
        input: Tensor output from the previous decoder layer.
        skip_connection: Tensor output from the corresponding encoder layer.
        num_filters: Number of filters or 'width' for the layer.

    Returns:
        Processed input data from the convolutional block.
    '''
    output = layers.Conv2DTranspose(num_filters, 3, 2, padding='same')(input)
    output = layers.Concatenate([input, skip_connection])
    output = convolutional_block(input, num_filters)
    return output


def build_model():
    '''
    Builds and compiles the U-Net model.

    Returns:
        The complete keras Model object
    '''
    # Define image size
    inputs = layers.Input(shape=(128, 128, 3))

    # Encoding (left side of U-Net)
    encode1, pool1 = encoder_block(inputs, 64)
    encode2, pool2 = encoder_block(pool1, 128)
    encode3, pool3 = encoder_block(pool2, 256)
    encode4, pool4 = encoder_block(pool3, 512)

    # Bottleneck (middle of U-Net)
    bridge = convolutional_block(pool4, 1024)

    # Decoding (right side of U-Net)
    decode4 = decoder_block(bridge, encode4, 512)
    decode3 = decoder_block(decode4, encode3, 256)
    decode2 = decoder_block(decode3, encode2, 128)
    decode1 = decoder_block(decode2, encode1, 64)

    # Final output processing step
    outputs = layers.Conv2D(3, 1, padding='same', activation='softmax')(decode1)

    # Compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam,
                  loss=losses.BinaryCrossentropy,
                  metrics=['accuracy', metrics.MeanIoU(num_classes=2)])

    return model
