from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout


def downsampling_conv(input, kernel_init, filters):
    conv1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(input)
    conv2 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(conv1)
    output = MaxPooling3D((2, 2, 2))(conv2)

    return conv2, output


def upsampling_conv(input, skip, kernel_init, filters):
    output = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(input)
    output = concatenate([output, skip])
    output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)
    output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)

    return output


def bottleneck(input, kernel_init, filters):
    output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(input)
    output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)

    return output


def unet(img_height, img_width, img_depth, img_channels, num_classes):
    input = Input((img_height, img_width, img_depth, img_channels))
    kernel_init = 'he_uniform'  # he_normal

    # Downampling path
    skip1, down1 = downsampling_conv(input, kernel_init, filters=16)
    skip2, down2 = downsampling_conv(down1, kernel_init, filters=32)
    skip3, down3 = downsampling_conv(down2, kernel_init, filters=64)
    skip4, down4 = downsampling_conv(down3, kernel_init, filters=128)
    
    # Bottleneck
    out = bottleneck(down4, kernel_init, filters=256)

    # Upsampling path
    up1 = upsampling_conv(out, skip4, kernel_init, filters=128)
    up2 = upsampling_conv(up1, skip3, kernel_init, filters=64)
    up3 = upsampling_conv(up2, skip2, kernel_init, filters=32)
    up4 = upsampling_conv(up3, skip1, kernel_init, filters=16)

    # Outputs
    if num_classes == 1:
        # sigmoid + binary crossentropy (1 class + background)
        output = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(up4)
        print('using sigmoid')
    else:
        # softmax + categorical crossentropy (2+ classes)
        output = Conv3D(num_classes, (1, 1, 1), activation='softmax')(up4)
        print('using softmax')

    model = Model(inputs=[input], outputs=[output])
    print("input shape: ", model.input_shape)
    print("output shape: ", model.output_shape)

    return model

if __name__ == '__main__':
    model = unet(128, 128, 128, 3, 2)
    model.summary()
