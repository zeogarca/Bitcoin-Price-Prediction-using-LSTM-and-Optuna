# Bitcoin-Price-Prediction-using-LSTM-and-Optuna

## Descripción del Proyecto
Este proyecto utiliza una red neuronal LSTM (Long Short-Term Memory) optimizada con un algoritmo genético (mediante Optuna) para predecir el precio de Bitcoin. El modelo se entrena con datos históricos del precio de cierre de Bitcoin y utiliza técnicas de aprendizaje automático para mejorar la precisión de las predicciones.

## Características Principales
- **Preprocesamiento de Datos**: Normalización de los datos y creación de secuencias para el modelo LSTM.
- **Optimización de Hiperparámetros**: Uso de Optuna para encontrar los mejores hiperparámetros (tamaño de secuencia, tamaño de capa oculta, tasa de dropout, etc.).
- **Modelo LSTM**: Implementación de una red LSTM para capturar patrones temporales en los datos.

## Estructura del Código

      Carga y Preprocesamiento de Datos:
      
      Se carga el dataset desde un archivo CSV.
      
      Se normalizan los datos utilizando MinMaxScaler.
      
      Se crean secuencias de datos para el entrenamiento del modelo.

## Definición del Modelo LSTM:

- La clase BitcoinLSTM define la arquitectura de la red LSTM, incluyendo capas LSTM y una capa lineal para la salida.

## Optimización con Optuna:

- La función objective define el proceso de entrenamiento y evaluación del modelo para cada conjunto de hiperparámetros propuestos por Optuna.

- Se utiliza la técnica de "poda" (pruning) para descartar combinaciones de hiperparámetros poco prometedoras.

## Entrenamiento y Evaluación:

- Se entrena el modelo con los mejores hiperparámetros encontrados.

- Se evalúa el modelo en el conjunto de validación y se visualizan los resultados.

## Uso
- Ejecución del Código:

                  El código está diseñado para ejecutarse en Google Colab, pero puede adaptarse para ejecutarse localmente.
                  
                  Asegúrate de tener el archivo de datos (btcusd_1-min_data.csv) en la ruta especificada.

## Personalización:

- Modifica los rangos de hiperparámetros en la función objective para ajustar la búsqueda según tus necesidades.

- Cambia el número de trials (n_trials) en study.optimize para aumentar o reducir la exploración del espacio de hiperparámetros.

## Resultados
- El proyecto genera un gráfico que compara los valores reales del precio de Bitcoin con las predicciones del modelo. Además, se imprimen los mejores hiperparámetros encontrados durante la optimización.

## Ejemplo de Salida
                  🔍 Mejores hiperparámetros:
                  {
                      'seq_len': 30,
                      'hidden_size': 64,
                      'num_layers': 2,
                      'dropout': 0.2,
                      'lr': 0.001,
                      'batch_size': 64
                  }
## Resultados

![image](https://github.com/user-attachments/assets/8d9f4868-9c13-4225-b0ce-0ba1464bf4d0)


## Contribuciones
Las contribuciones son bienvenidas. Si encuentras algún error o tienes sugerencias para mejorar el modelo, no dudes en abrir un issue o enviar un pull request.
