import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import json
import os
import seaborn as sns


def evaluate(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) # ajouter si besoin le paramètre : display_labels=ordinal_values
    disp.plot()
    plt.show()

    # Generate the classification report as a dictionary
    report_dict = (classification_report(y_test, y_pred, output_dict=True))

    # Convert the dictionary to a JSON string
    report_json = json.dumps(report_dict, indent=4)

    # Save the JSON string to a file
    with open('classification_report.json', 'w') as json_file:
        json_file.write(report_json)

    # Define experiment directories
    experiment_dir = "experiments"
    data_dir = os.path.join(experiment_dir, "data")
    metrics_dir = os.path.join(experiment_dir, "metrics")

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Define file names
    filename = (
        f"{'_'.join(model_name.split('/'))}"
        f"_accuracy_{round(accuracy * 100)}"
        f"_dt_{timestamp.replace(':', '').replace('-', '')}"
    )

    filename_meta = os.path.join(metrics_dir, filename + ".json")
    filename_data = os.path.join(data_dir, filename + ".csv")
    print(filename_meta)
    print(filename_data)


    print("Classification report saved to 'classification_report.json'")
    print("✅ Evaluation done")



def global_model_comparison():
    plt.figure(figsize=(5, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test)
        )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
