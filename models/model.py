from tensorflow.keras import layers, models, regularizers

def create_model(config):
    kernel_size = config.kernel_size
    regularizer = regularizers.l2(config.reg)
    dropout_rate = config.dropout
    size_0 = config.start_filters

    # Input layer
    in_data = layers.Input(shape=(1024, 1))

    # Encoder part
    conv0 = layers.Conv1D(size_0, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(in_data)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Conv1D(size_0, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    pool0 = layers.MaxPooling1D(pool_size=2)(conv0)

    size_1 = size_0 * 2
    conv1 = layers.Conv1D(size_1, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(pool0)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv1D(size_1, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout_rate)(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(conv1)

    size_2 = size_1 * 2
    conv2 = layers.Conv1D(size_2, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv1D(size_2, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = layers.BatchNormalization()(conv2)

    # Decoder part
    up1 = layers.UpSampling1D(size=2)(conv2)
    up_conv1 = layers.Conv1D(size_2, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    up_conv1 = layers.BatchNormalization()(up_conv1)
    merge1 = layers.concatenate([conv1, up_conv1], axis=2)
    conv3 = layers.Conv1D(size_1, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv1D(size_1, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(dropout_rate)(conv3)

    up2 = layers.UpSampling1D(size=2)(conv3)
    up_conv2 = layers.Conv1D(size_1, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    up_conv2 = layers.BatchNormalization()(up_conv2)
    merge2 = layers.concatenate([conv0, up_conv2], axis=2)
    conv4 = layers.Conv1D(size_0, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv1D(size_0, kernel_size, kernel_regularizer=regularizer, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.BatchNormalization()(conv4)

    # Output layer
    out_data = layers.Conv1D(1, kernel_size, activation='sigmoid', padding='same')(conv4)

    model = models.Model(inputs=[in_data], outputs=[out_data])
    model.summary()

    return model

class ModelConfig:
    def __init__(self, kernel_size, reg, dropout, start_filters):
        self.kernel_size = kernel_size
        self.reg = reg
        self.dropout = dropout
        self.start_filters = start_filters
