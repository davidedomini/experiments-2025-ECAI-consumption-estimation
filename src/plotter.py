import glob
import matplotlib
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


def beutify_model_name(model):
    if "sparsity-0.0" in model:
        model_type = "0.0"
    elif "sparsity-0.3" in model:
        model_type = "0.3"
    elif "sparsity-0.5" in model:
        model_type = "0.5"
    elif "sparsity-0.7" in model:
        model_type = "0.7"
    elif "sparsity-0.9" in model:
        model_type = "0.9"
    else:
        model_type = "Unknown"
    return model_type

def plot(data, metric):
    # sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="Algorithm", y=metric, palette="viridis",width=1)

    plt.xlabel("$\psi$")
    ax.set_axisbelow(True)
    if 'Time' in metric:
        m = '$Mean \: Time \: (s)$'
    else:
        m = '$Mean \: Accuracy$'
    plt.ylabel(m)
    plt.xticks(rotation=45)  
    if 'Accuracy' in metric:
        plt.ylim(0.7, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    metric = metric.replace(" ", "")
    plt.tight_layout()
    plt.savefig(f'charts/{metric}.pdf', dpi=300)
    plt.close()


if __name__ == '__main__':

        
    data_output_directory = Path('charts')
    data_output_directory.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update({'axes.titlesize': 20})
    matplotlib.rcParams.update({'axes.labelsize': 45})
    matplotlib.rcParams.update({'xtick.labelsize': 35})
    matplotlib.rcParams.update({'ytick.labelsize': 35})
    plt.rcParams.update({"text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    csv_files = glob.glob("data/inference-time*.csv") 

    times = {
        "0.0": [],
        "0.3": [],
        "0.5": [],
        "0.7": [],
        "0.9": []
    }

    accuracies = {
        "0.0": [],
        "0.3": [],
        "0.5": [],
        "0.7": [],
        "0.9": []
    }

    for file in csv_files:
        model_type = beutify_model_name(file)
        df = pd.read_csv(file)
        times[model_type].append(df['Time'].mean())
        accuracies[model_type].append(df['Accuracy'].mean())


    result_time = [{"Algorithm": algo, "Mean Time (s)": sum(times) / len(times)} for algo, times in times.items()]
    result_acc = [{"Algorithm": algo, "Mean Accuracy": sum(acc) / len(acc)} for algo, acc in accuracies.items()]

    df_time = pd.DataFrame(result_time).sort_values(by="Algorithm", ascending=True)
    df_acc = pd.DataFrame(result_acc).sort_values(by="Algorithm", ascending=True)
    plot(df_time, "Mean Time (s)")
    plot(df_acc, "Mean Accuracy")

