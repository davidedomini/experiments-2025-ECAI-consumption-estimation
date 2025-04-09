import glob
import matplotlib
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


def extract_sparsity(model):
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


    ################# PLOT INFERENCE TIME AND ACCURACY #################

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
        sparsity = extract_sparsity(file)
        df = pd.read_csv(file)
        times[sparsity].append(df['Time'].mean())
        accuracies[sparsity].append(df['Accuracy'].mean())


    result_time = [{"Algorithm": algo, "Mean Time (s)": sum(times) / len(times)} for algo, times in times.items()]
    result_acc = [{"Algorithm": algo, "Mean Accuracy": sum(acc) / len(acc)} for algo, acc in accuracies.items()]

    df_time = pd.DataFrame(result_time).sort_values(by="Algorithm", ascending=True)
    df_acc = pd.DataFrame(result_acc).sort_values(by="Algorithm", ascending=True)
    plot(df_time, "Mean Time (s)")
    plot(df_acc, "Mean Accuracy")


    ################# PLOT ENERGY CONSUMPTION #################

    consumptions = {
        "0.0": [],
        "0.3": [],
        "0.5": [],
        "0.7": [],
        "0.9": []
    }

    csv_files = glob.glob("data/inference-energy*False.csv")

    for file in csv_files:
        sparsity = extract_sparsity(file)
        df = pd.read_csv(file, sep=';')
        consumptions[sparsity].append(df['package_0'].mean())

    print(consumptions)
    
    experiments = list(consumptions.keys())
    consumption_values = [v[0] for v in consumptions.values()]

    consumptions = pd.DataFrame({
        'Experiment': experiments,
        'Consumption': consumption_values
    })
    initial_consumption = consumptions['Consumption'].iloc[0]
    consumptions['Relative Consumption'] = consumptions['Consumption'] / initial_consumption

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Experiment', y='Relative Consumption', data=consumptions, palette='rocket')

    plt.title("Relative Energy Consumption (w.r.t. $\psi = 0.0$)", fontsize=16)
    plt.ylabel("Relative  consumption")
    plt.xlabel("$\psi$")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('charts/relative_engergy_consumption.pdf', dpi=300)