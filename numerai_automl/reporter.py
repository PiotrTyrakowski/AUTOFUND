# numerai_automl/reporter.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Reporter:
    def __init__(self, scores: pd.DataFrame):
        self.scores = scores

    def generate_summary(self) -> pd.DataFrame:
        summary = self.scores.describe().T
        summary = summary.rename(columns={
            "mean": "Average Score",
            "std": "Score Std Dev",
            "50%": "Median Score",
            "min": "Minimum Score",
            "max": "Maximum Score"
        })
        return summary

    def plot_cumulative_scores(self):
        plt.figure(figsize=(10,6))
        for column in self.scores.columns:
            plt.plot(self.scores[column].cumsum(), label=column)
        plt.title("Cumulative Correlation Over Time")
        plt.xlabel("Era")
        plt.ylabel("Cumulative CORR")
        plt.legend()
        plt.show()

    def plot_score_distribution(self):
        plt.figure(figsize=(10,6))
        for column in self.scores.columns:
            sns.kdeplot(self.scores[column], label=column, shade=True)
        plt.title("Distribution of Scores")
        plt.xlabel("Correlation Score")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def save_report(self, filename: str = "report.pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(filename) as pdf:
            plt.figure(figsize=(10,6))
            for column in self.scores.columns:
                plt.plot(self.scores[column].cumsum(), label=column)
            plt.title("Cumulative Correlation Over Time")
            plt.xlabel("Era")
            plt.ylabel("Cumulative CORR")
            plt.legend()
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(10,6))
            for column in self.scores.columns:
                sns.kdeplot(self.scores[column], label=column, shade=True)
            plt.title("Distribution of Scores")
            plt.xlabel("Correlation Score")
            plt.ylabel("Density")
            plt.legend()
            pdf.savefig()
            plt.close()
