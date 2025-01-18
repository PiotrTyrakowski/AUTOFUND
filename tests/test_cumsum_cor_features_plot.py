import pandas as pd
import numpy as np
from numerai_automl.visual.cumsum_cor_features_plot import CumsumCorrFeaturesPlot


def test_cumsum_cor_features_plot():
    data = {
        'feature_elemental_easier_alkalinity': [-0.009335, 0.008002, 0.003275, 0.002286, 0.009947],
        'feature_gowned_undiluted_islay': [0.010063, -0.005904, -0.013445, -0.013983, -0.000354],
        'feature_crashing_lubberly_sanitarium': [-0.007345, -0.010797, -0.020163, -0.018314, -0.007507]
    }

    # Indeksy
    index = ['0001', '0005', '0009', '0013', '0017']

    # Tworzenie ramki danych
    df = pd.DataFrame(data, index=index)
    df.index.name = 'era'
    print("DATA:")
    print(df)
    print("LINE PLOT:")
    rp = CumsumCorrFeaturesPlot(df)
    fig = rp.get_plot()

    fig.show()


if __name__ == '__main__':
    test_cumsum_cor_features_plot()
