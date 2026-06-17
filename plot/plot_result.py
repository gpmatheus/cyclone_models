"""
Script para visualizar resultados de predições da rede neural em dados de teste.
Plota scatter plot das predições vs valores reais com diagonal de referência.
"""

import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error

keras = tensorflow.keras


class PredictionVisualizer:
    """Classe para carregar modelo, fazer predições e visualizar resultados."""
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        img_width: int = 64,
        rotations: int = 10,
        channels: list = None,
    ):
        """
        Inicializa o visualizador.
        
        Args:
            model_path: Caminho para o modelo treinado
            test_data_path: Caminho para dados de teste (já normalizado)
            img_width: Largura da imagem
            rotations: Número de rotações a aplicar
            channels: Canais a utilizar
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.img_width = img_width
        self.rotations = rotations
        self.channels = channels or [0, 3]
        self.angles = tf.cast(tf.linspace(0, 360, self.rotations), tf.float32)
        
        self.model = None
        self.images = None
        self.info = None
        
    def load_model(self) -> None:
        """Carrega o modelo treinado."""
        print(f"Carregando modelo de {self.model_path}...")
        self.model = keras.models.load_model(self.model_path, compile=False)
        self.model.compile(
            optimizer='adam',
            loss=keras.losses.MeanSquaredError(),
            metrics=['mse']
        )
        print("Modelo carregado com sucesso!")
        

    def load_test_data(self) -> None:
        """Carrega dados de teste do arquivo HDF5."""
        print(f"Carregando dados de teste de {self.test_data_path}...")
        with h5py.File(self.test_data_path, mode='r') as file:
            self.images = file['matrix'][:]
            # self.info = file['info'][:]
        
        self.info = pd.read_hdf(self.test_data_path, key="info", mode="r")["Vmax"]
        print(f"Dados carregados! Total de amostras: {len(self.images)}")
        
    def preprocess_image_tf(self, image: tf.Tensor, angle_rad: float) -> tf.Tensor:
        """
        Aplica rotação a uma imagem usando transformação afim.
        
        Args:
            image: Tensor de imagem
            angle_rad: Ângulo em radianos
            
        Returns:
            Imagem rotacionada e redimensionada
        """
        angle_rad = angle_rad * np.pi / 180.0
        image_shape = tf.shape(image)[0:2]
        
        cx = tf.cast(image_shape[1] / 2, tf.float32)
        cy = tf.cast(image_shape[0] / 2, tf.float32)
        
        cos_a = tf.math.cos(angle_rad)
        sin_a = tf.math.sin(angle_rad)
        
        transform = tf.stack([
            cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy,
            sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx,
            0.0,    0.0
        ])
        transform = tf.reshape(transform, [8])
        transform = tf.expand_dims(transform, 0)
        
        image = tf.expand_dims(image, 0)
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=image,
            transforms=transform,
            output_shape=image_shape,
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0
        )
        rotated = tf.squeeze(rotated, 0)
        return tf.image.resize_with_crop_or_pad(rotated, self.img_width, self.img_width)
        
    def parse_example(self, image: tf.Tensor) -> tf.Tensor:
        """
        Processa uma imagem com múltiplas rotações.
        
        Args:
            image: Tensor de imagem
            
        Returns:
            Stack de imagens rotacionadas
        """
        image = tf.cast(image, tf.float32)
        image = tf.convert_to_tensor([
            self.preprocess_image_tf(image, ang) for ang in self.angles
        ])
        return image
        
    def predict_single(self, image: tf.Tensor) -> float:
        """
        Faz predição para uma única imagem.
        
        Args:
            image: Tensor de imagem
            
        Returns:
            Predição (média sobre as rotações)
        """
        res = self.model.predict(self.parse_example(image), verbose=0)
        return float(tf.math.reduce_mean(res))
        
    def predict_all(self) -> np.ndarray:
        """
        Faz predições para todas as imagens.
        
        Returns:
            Array com as predições
        """
        print("Fazendo predições...")
        predictions = tf.map_fn(
            self.predict_single,
            elems=self.images,
            fn_output_signature=tf.float32
        )
        print("Predições concluídas!")
        return predictions.numpy()
        
    def plot_results(self, predictions: np.ndarray, save_path: str = None) -> None:
        """
        Plota scatter plot com linha de referência y=x.
        
        Args:
            predictions: Array com as predições
            save_path: Caminho opcional para salvar a figura
        """
        real_values = self.info
        
        # Calcula métricas
        mse = mean_squared_error(real_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_values, predictions)
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Scatter plot
        ax.scatter(real_values, predictions, alpha=0.6, s=30, color='blue', label='Predições')
        
        # Linha de referência y=x
        min_val = min(real_values.min(), predictions.min())
        max_val = max(real_values.max(), predictions.max())
        margin = (max_val - min_val) * 0.05
        lim_min = min_val - margin
        lim_max = max_val + margin
        
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 
                color='red', linestyle='--', linewidth=2, label='y=x (predição perfeita)')
        
        # Configurações dos eixos
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')
        
        # Labels e título
        ax.set_xlabel("Velocidade do vento real (nós)", fontsize=12)
        ax.set_ylabel("Velocidade do vento predita (nós)", fontsize=12)
        title = f"Predições vs Valores Reais\nRMSE: {rmse:.2f} nós | MAE: {mae:.2f} nós"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            print(f"Salvando figura em {save_path}...")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.savefig("resultado.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprime métricas
        print("\n" + "="*50)
        print("MÉTRICAS DE DESEMPENHO")
        print("="*50)
        print(f"MSE (Erro Quadrático Médio): {mse:.4f}")
        print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f} nós")
        print(f"MAE (Erro Absoluto Médio): {mae:.4f} nós")
        print(f"Total de amostras: {len(real_values)}")
        print("="*50 + "\n")
        
    def run(self, save_plot_path: str = None) -> np.ndarray:
        """
        Executa todo o pipeline: carrega dados, faz predições e plota.
        
        Args:
            save_plot_path: Caminho opcional para salvar a figura
            
        Returns:
            Array com as predições
        """
        self.load_model()
        self.load_test_data()
        
        predictions = self.predict_all()
        self.plot_results(predictions, save_plot_path)
        
        return predictions


def main(img_width=64):
    """Função principal com configurações do experimento."""
    
    # ==================== CONFIGURAÇÕES ====================
    
    # Caminhos
    model_path = os.getenv("MODEL_PATH") or "model.keras"
    test_data_path = os.getenv("PREPROCESSED_TEST_PATH") or "data/preprocessed/test.h5"
    save_plot_path = os.getenv("SAVE_PLOT_PATH") or "plot/results_plot.png"  # Opcional: descomentar para salvar
    
    # Parâmetros do modelo
    rotations = 10
    channels = [0, 3]
    
    # =========================================================
    
    # Cria visualizador e executa
    visualizer = PredictionVisualizer(
        model_path=model_path,
        test_data_path=test_data_path,
        img_width=img_width,
        rotations=rotations,
        channels=channels,
    )
    
    predictions = visualizer.run(save_plot_path=save_plot_path)
    
    return predictions


if __name__ == "__main__":
    main()