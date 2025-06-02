from tensorflow.keras import layers, models

class AutoencoderModel:
    """簡單的自編碼器，用於深度特徵提取"""
    def __init__(self, input_dim, latent_dim=3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._build_model()

    def _build_model(self):
        inp = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(inp)
        encoded = layers.Dense(self.latent_dim, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        self.autoencoder = models.Model(inp, decoded)
        self.encoder = models.Model(inp, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, data, epochs=50, batch_size=16):
        """訓練自編碼器"""
        self.autoencoder.fit(data, data,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             verbose=0)

    def extract_features(self, data):
        """使用編碼器提取潛在特徵"""
        return self.encoder.predict(data)
