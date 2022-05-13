from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
import tensorflow as tf

class Unet(tf.keras.Model):
    def __init__(self, img_height, img_width, img_depth, img_channels, num_classes, name=None):
        super().__init__()
        self.kernel_init = 'he_uniform'  # he_normal
        
        # Input
        self.inputs = Input((img_height, img_width, img_depth, img_channels))

         # Output
        if num_classes == 1:
            # sigmoid + binary crossentropy (1 class + background)
            self.outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')
            print('using sigmoid')
        else:
            # softmax + categorical crossentropy (2+ classes)
            self.outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')
            print('using softmax')

    def downsampling_conv(self, input, kernel_init, filters):
        conv1 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(input)
        conv2 = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(conv1)
        output = MaxPooling3D((2, 2, 2))(conv2)

        return conv2, output

    def upsampling_conv(self, input, skip, kernel_init, filters):
        output = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(input)
        output = concatenate([output, skip])
        output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)
        output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)

        return output

    def bottle_neck(self, input, kernel_init, filters):
        output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(input)
        output = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_init, padding='same')(output)

        return output

    def call(self, inputs):
        # Downsampling path
        skip1, down1 = self.downsampling_conv(inputs, self.kernel_init, filters=16)
        skip2, down2 = self.downsampling_conv(down1, self.kernel_init, filters=32)
        skip3, down3 = self.downsampling_conv(down2, self.kernel_init, filters=64)
        skip4, down4 = self.downsampling_conv(down3, self.kernel_init, filters=128)

        # Bottleneck
        out = self.bottle_neck(down4, self.kernel_init, filters=256)

        # Upsampling path
        up1 = self.upsampling_conv(out, skip4, self.kernel_init, filters=128)
        up2 = self.upsampling_conv(up1, skip3, self.kernel_init, filters=64)
        up3 = self.upsampling_conv(up2, skip2, self.kernel_init, filters=32)
        up4 = self.upsampling_conv(up3, skip1, self.kernel_init, filters=16)

        return self.outputs(up4)


if __name__ == '__main__':
    model = Unet(128, 128, 128, 3, 2)
    print(model)
    m = model.compile()
    print(m)
    #model.summary()
