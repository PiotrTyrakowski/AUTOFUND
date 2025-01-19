from numerai_automl.raport_manager.raport_manager import RaportManager
from numerai_automl.visual.cumsum_cor_plot import CumSumCorPlot
import pandas as pd
from numerai_automl.visual.radar_plot import RadarPlot

def test_cumsum_cor_plot():
    df = pd.read_csv("return_data.csv")
    df = df[["predictions_model_target", "neutralized_predictions_model_target", "predictions_model_omega", "predictions_model_meta_weighted", "predictions_model_meta_lgbm", 'era',  "target",]]
    csp = CumSumCorPlot(df)

    fig = csp.get_plot()


    rm = RaportManager([fig])
    rm.generate_html("cumsum_cor_plot.html")


    

    # fig.show()
    # fig.savefig('cumsum_cor_plot.png')

def test_radar_plot():
    df = pd.read_csv("return_data_for_scoring.csv")
    rp = RadarPlot(df)
    fig = rp.get_plot()
    rm = RaportManager([fig])
    rm.generate_html("radar_plot.html")

if __name__ == "__main__":
    test_radar_plot()