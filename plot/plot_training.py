import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("history.pkl", "rb") as file:
    history = pickle.load(file)

mi = min(history["val_mse"])
print(f"min: {mi}")

step = 1

plt.plot(np.array(list(range(len(history["mse"]))))[::step], history["mse"][::step], color='blue', label='Erro do Treinamento')
plt.plot(np.array(list(range(len(history["val_mse"]))))[::step], history["val_mse"][::step], color='red', label='Erro de Validação')

plt.legend()

plt.show()