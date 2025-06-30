import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    auc,
    RocCurveDisplay,
)


def make_date_format(date: str):
    date_list = list(date)
    date_list.insert(0, "20")
    date_list.insert(3, "-")
    date_list.insert(6, "-")
    date_list.insert(12, ":")
    date_list.insert(15, ":")
    date_list[18] = "."
    return "".join(date_list)


def read_hdfs(path_to_dataset: str) -> list[str]:
    data = []
    row_string = " "
    with open(path_to_dataset, encoding="utf-8", errors="ignore") as file:
        while row_string != "":
            row_string = file.readline()
            if len(row_string) > 1:
                splitted_log = row_string.replace("\n", "").split(" ")
                date_str = " ".join(splitted_log[:3])
                date_str = make_date_format(date_str)
                tag = splitted_log[3]
                log_text = " ".join(splitted_log[4:])
                data.append(" ".join([date_str, tag, log_text]))
    return data


def read_logs(path_to_dataset: str, dataset_type: str):
    if dataset_type == "HDFS":
        return read_hdfs(path_to_dataset)


def visual_anomaly_count(dataset: pd.DataFrame):
    dataset["TimeStamp_"] = dataset["TimeStamp"].values.astype("datetime64[m]")
    fig = go.Figure()
    grouped_df = dataset.groupby("TimeStamp_")["Label"].count()
    fig.add_trace(
        go.Scatter(
            x=grouped_df.index,
            y=grouped_df.values,
            mode="lines",
            line_color="blue",
            line_width=0.7,
            opacity=0.8,
            name="Number of logs per interval",
            showlegend=True,
        )
    )
    grouped_df = (
        dataset[dataset["Label"] == "Anomaly"].groupby("TimeStamp_")["Label"].count()
    )
    fig.add_trace(
        go.Scatter(
            x=grouped_df.index,
            y=grouped_df.values,
            mode="lines",
            line_color="red",
            line_width=0.7,
            opacity=0.8,
            name="Number of anomaly logs per interval",
            showlegend=True,
        )
    )
    fig.update_layout(
        title="Logs statistics",
        yaxis_title="Number of logs",
        font=dict(size=16),
        legend=dict(orientation="h"),
    )
    fig.show()


def find_long_alerts(
    score: pd.Series, treshold: float, window_width: int, min_condition: int
) -> pd.Series:
    indication = score.rolling(window_width).apply(
        lambda x: 1 if sum(x >= treshold) >= min_condition else 0
    )
    return score.loc[indication == 1]


def visual_mean_dist_to_nbrs(
    score,
    treshold_score: float = 0.1,
    window_width: int = 40,
    min_condition: int = 20,
    y_log: bool = False,
    roll_window: int = None,
    return_fig: bool = True,
    font_size: int = 20,
):
    if roll_window is not None:
        score = pd.Series(score).rolling(roll_window).median().copy()
    ind_faults = score >= treshold_score
    long_alerts = find_long_alerts(
        score, treshold_score, window_width=window_width, min_condition=min_condition
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score.index,
            y=score,
            mode="lines",
            line_color="lightseagreen",
            line_width=2,
            opacity=0.7,
            name="Score",
        )
    )
    fig.add_hline(
        y=treshold_score,
        line_width=2,
        opacity=0.6,
        line_color="red",
        annotation_text="Upper control limit",
        annotation_position="top right",
    )
    if y_log:
        y_markers = 1
    else:
        y_markers = 0
    fig.add_trace(
        go.Scatter(
            x=ind_faults.loc[ind_faults].index,
            y=np.full(ind_faults.sum(), y_markers),
            mode="markers",
            marker_symbol="diamond",
            line_color="lightcoral",
            opacity=0.6,
            name="Alerts",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=long_alerts.index,
            y=np.full(len(long_alerts), y_markers),
            mode="markers",
            marker_symbol="diamond",
            line_color="red",
            opacity=0.6,
            name=f"Long alerts ({min_condition} out of {window_width} alerts)",
            showlegend=True,
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Anomaly score",
        font=dict(size=font_size),
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(range=(score.index.min(), score.index.max()), constrain="domain")
    if y_log:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(range=(-treshold_score * 0.2, np.quantile(score, 0.9975)))
    if return_fig:
        return fig
    else:
        fig.show()


def calc_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series, plot_roc: bool = True
):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    roc_auc = roc_auc_score(y_true, y_score)

    print("FPR = %.3f, TPR = %.3f, ROC AUC = %.3f" % (fpr, tpr, roc_auc))

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name="ROC curve"
        )
        display.plot()
        plt.show()
