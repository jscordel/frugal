import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate(y_test, y_pred):
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) # ajouter si besoin le paramètre : display_labels=ordinal_values
    disp.plot()
    plt.show()

    # Scores
    print(classification_report(y_test, y_pred))
