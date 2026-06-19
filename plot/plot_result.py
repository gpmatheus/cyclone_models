"""
Script para visualizar resultados de predições da rede neural em dados de teste.
Plota scatter plot das predições vs valores reais com diagonal de referência.

Utiliza Test-Time Augmentation (TTA) por Rotation Blending: cada imagem de teste
é submetida ao modelo em múltiplas rotações e as predições são agregadas por média,
reduzindo a sensibilidade à orientação arbitrária do ciclone na imagem.

Para que o RMSE/MAE do gráfico (Tabela 3) seja comparável ao usado no Teste de
Wilcoxon (Tabela 4), ambos devem ser calculados sobre o mesmo subconjunto de
amostras. O subconjunto alinhado corresponde à remoção das primeiras N imagens de
cada ciclone, onde N depende do pré-processamento do modelo:

    original     : REMOVE_FROM_TC = 2
    tcc_I (Diff1): REMOVE_FROM_TC = 1
    tcc_II (Diff2): REMOVE_FROM_TC = 0
    resnet        : REMOVE_FROM_TC = 0
    mobilenet_v2  : REMOVE_FROM_TC = 0

Variáveis de ambiente:
  MODEL_TYPE            : Tipo de modelo — determina REMOVE_FROM_TC automaticamente
  MODEL_PATH            : Caminho para o modelo treinado (.keras)
  PREPROCESSED_TEST_PATH: Caminho para o arquivo HDF5 de teste pré-processado
  SAVE_PLOT_PATH        : Caminho de saída para o gráfico
  REMOVE_FROM_TC        : Sobrescreve o valor automático se definido
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

# ---------------------------------------------------------------------------
# Configuração: mesma lógica de model_error_analysis.py
# ---------------------------------------------------------------------------

_REMOVE_MAP = {
    'original':     2,
    'tcc_I':        1,
    'tcc_II':       0,
    'resnet':       0,
    'mobilenet_v2': 0,
}

MODEL_TYPE = os.getenv('MODEL_TYPE') or 'original'
REMOVE_FROM_TC = int(
    os.getenv('REMOVE_FROM_TC') or _REMOVE_MAP.get(MODEL_TYPE, 0)
)

print(f"MODEL_TYPE     : {MODEL_TYPE}")
print(f"REMOVE_FROM_TC : {REMOVE_FROM_TC}  (imagens removidas do início de cada ciclone)")


class PredictionVisualizer:
    """Carrega modelo, aplica Rotation Blending (TTA) e plota predições vs. reais."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        img_width: int = 64,
        rotations: int = 10,
    ):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.img_width = img_width
        self.rotations = rotations
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

    def load_test_data(self, remove_from_tc: int = 0) -> None:
        """Carrega dados de teste do arquivo HDF5.

        Se remove_from_tc > 0, descarta as primeiras N imagens de cada ciclone
        (ordenando por ID e tempo), garantindo que o conjunto de avaliação seja
        o mesmo subconjunto alinhado usado no Teste de Wilcoxon.
        """
        print(f"Carregando dados de teste de {self.test_data_path}...")
        with h5py.File(self.test_data_path, mode='r') as file:
            all_images = file['matrix'][:]

        info_df = pd.read_hdf(self.test_data_path, key="info", mode="r")
        info_df = info_df.reset_index(drop=True)

        if remove_from_tc > 0 and 'ID' in info_df.columns and 'time' in info_df.columns:
            info_sorted = info_df.sort_values(['ID', 'time'])
            kept = (
                info_sorted
                .groupby('ID', group_keys=False)
                .apply(lambda x: x.iloc[remove_from_tc:])
            )
            keep_idx = kept.index.to_numpy()
            self.images = all_images[keep_idx]
            self.info = info_df.loc[keep_idx, 'Vmax'].reset_index(drop=True)
            print(f"  Filtro aplicado: {len(all_images)} → {len(self.images)} amostras "
                  f"({remove_from_tc} imagem(ns) removida(s) por ciclone)")
        else:
            self.images = all_images
            self.info = info_df['Vmax']

        print(f"Total de amostras para predição: {len(self.images)}")

    def preprocess_image_tf(self, image: tf.Tensor, angle_rad: float) -> tf.Tensor:
        """Aplica rotação a uma imagem usando transformação afim."""
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
        """Gera stack de imagens rotacionadas (TTA)."""
        image = tf.cast(image, tf.float32)
        image = tf.convert_to_tensor([
            self.preprocess_image_tf(image, ang) for ang in self.angles
        ])
        return image

    def predict_single(self, image: tf.Tensor) -> float:
        """Predição com TTA: média das predições sobre todas as rotações."""
        res = self.model.predict(self.parse_example(image), verbose=0)
        return float(tf.math.reduce_mean(res))

    def predict_all(self) -> np.ndarray:
        """Faz predições para todas as imagens."""
        print("Fazendo predições (TTA — Rotation Blending)...")
        predictions = tf.map_fn(
            self.predict_single,
            elems=self.images,
            fn_output_signature=tf.float32
        )
        print("Predições concluídas!")
        return predictions.numpy()

    def plot_results(self, predictions: np.ndarray, save_path: str = None) -> None:
        """Plota scatter plot com linha de referência y=x."""
        real_values = self.info

        mse  = mean_squared_error(real_values, predictions)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(real_values, predictions)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        ax.scatter(real_values, predictions, alpha=0.6, s=30, color='blue', label='Predições')

        min_val = min(real_values.min(), predictions.min())
        max_val = max(real_values.max(), predictions.max())
        margin  = (max_val - min_val) * 0.05
        lim_min = min_val - margin
        lim_max = max_val + margin

        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color='red', linestyle='--', linewidth=2, label='y=x (predição perfeita)')

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')

        ax.set_xlabel("Velocidade do vento real (nós)", fontsize=12)
        ax.set_ylabel("Velocidade do vento predita (nós)", fontsize=12)
        ax.set_title(
            f"Predições vs Valores Reais\nRMSE: {rmse:.2f} nós | MAE: {mae:.2f} nós",
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            print(f"Salvando figura em {save_path}...")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        print("\n" + "="*50)
        print("MÉTRICAS DE DESEMPENHO")
        print("="*50)
        print(f"MSE  (Erro Quadrático Médio):        {mse:.4f}")
        print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f} nós")
        print(f"MAE  (Erro Absoluto Médio):           {mae:.4f} nós")
        print(f"Total de amostras: {len(real_values)}")
        print("="*50 + "\n")

    def run(self, save_plot_path: str = None, remove_from_tc: int = 0) -> np.ndarray:
        """Pipeline completo: carrega dados, faz predições e plota."""
        self.load_model()
        self.load_test_data(remove_from_tc=remove_from_tc)

        predictions = self.predict_all()
        self.plot_results(predictions, save_plot_path)

        return predictions


def main(img_width=64):
    """Função principal com configurações do experimento."""
    model_path     = os.getenv("MODEL_PATH")             or "model.keras"
    test_data_path = os.getenv("PREPROCESSED_TEST_PATH") or "data/preprocessed/test.h5"
    save_plot_path = os.getenv("SAVE_PLOT_PATH")         or "plot/results_plot.png"

    visualizer = PredictionVisualizer(
        model_path=model_path,
        test_data_path=test_data_path,
        img_width=img_width,
        rotations=10,
    )

    predictions = visualizer.run(
        save_plot_path=save_plot_path,
        remove_from_tc=REMOVE_FROM_TC,
    )
    return predictions


if __name__ == "__main__":
    main()
