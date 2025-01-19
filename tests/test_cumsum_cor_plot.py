from numerai_automl.scorer.scorer import Scorer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def test_cumsum_cor_plot():

    scorer = Scorer()

    # stwórz dataframe z danymi testowymi
    # era to nie będą liczby od 1 do 500 po 10 razy każda

    df = pd.DataFrame({
        'era': np.repeat(np.arange(1, 501), 10),
        # losowe wartości z [0,0.2, 0.5,0.8,1]
        'target': np.random.choice([0,0.2, 0.5,0.8,1], 5000)
    })
    df["prediction_m1"] = df['target'] + np.random.normal(0, 0.1, 5000)
    df["prediction_m2"] = df['target'] ** 2 + np.random.normal(0, 0.1, 5000)
    df["prediction_m3"] = df['target'] ** 2
    df["prediction_m4"] = np.random.choice([0,0.2, 0.5,0.8], 5000)
    df['main_model'] = np.random.choice([0,0.2, 0.5,0.8,1], 5000)
    #minmax scaling for predictions
    minmax=MinMaxScaler()
    df[['prediction_m1','prediction_m2','prediction_m3']]=minmax.fit_transform(df[['prediction_m1','prediction_m2','prediction_m3']])
    print("DATA input:")
    print(df)
    print("DATA before plotting:")
    df2=scorer.compute_cumsum_correlation_per_era(df,'target')
    print(df2)
    # Usuwamy prefiks "prediction_" z nazw kolumn
    df2.columns = [col.replace('prediction_', '') for col in df2.columns]

    # Tworzymy wykres
    df2.plot(figsize=(10, 6), linewidth=2)

    # Dostosowanie wykresu
    plt.title('Cumulative Correlation of Predictions and Target for each Era', fontsize=16)
    plt.xlabel('Era', fontsize=14)
    plt.ylabel('Cumulative Correlation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model', fontsize=12)
    plt.show()

    plt.savefig('cumsum_cor_plot.png')
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    test_cumsum_cor_plot()