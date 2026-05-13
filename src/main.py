from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================
# RUTAS DEL PROYECTO
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "clientes_credito.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================
# CARGA DE DATOS
# ==============================

def cargar_datos():

    df = pd.read_csv(DATA_PATH)

    print("\n=== Vista inicial del dataset ===")
    print(df.head())

    print("\n=== Información general ===")
    print(df.info())

    print("\n=== Estadísticas descriptivas ===")
    print(df.describe())

    return df


# ==============================
# REGRESIÓN LINEAL
# ==============================

def modelo_regresion_lineal(df):

    print("\n================ REGRESIÓN LINEAL ================")

    X = df[
        [
            "edad",
            "ingresos",
            "historial_crediticio",
            "deuda",
            "horas_trabajo_semana"
        ]
    ]

    y = df["gasto_mensual"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    modelo = LinearRegression()

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"R2 : {r2:.3f}")

    coeficientes = pd.DataFrame({
        "variable": X.columns,
        "coeficiente": modelo.coef_
    })

    print("\nCoeficientes del modelo:")
    print(coeficientes)

    plt.figure()

    plt.scatter(y_test, y_pred)

    plt.xlabel("Gasto mensual real")
    plt.ylabel("Gasto mensual predicho")

    plt.title("Regresión lineal")

    plt.savefig(
        OUTPUT_DIR / "regresion_lineal_real_vs_predicho.png",
        bbox_inches="tight"
    )

    plt.close()


# ==============================
# EVALUACIÓN DE CLASIFICACIÓN
# ==============================

def evaluar_clasificacion(nombre, y_test, y_pred):

    print(f"\n--- Evaluación: {nombre} ---")

    print(
        f"Accuracy : "
        f"{accuracy_score(y_test, y_pred):.3f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, y_pred, zero_division=0):.3f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, y_pred, zero_division=0):.3f}"
    )

    print(
        f"F1-score : "
        f"{f1_score(y_test, y_pred, zero_division=0):.3f}"
    )

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ==============================
# REGRESIÓN LOGÍSTICA
# ==============================

def modelo_regresion_logistica(df):

    print("\n================ REGRESIÓN LOGÍSTICA ================")

    X = df[
        [
            "edad",
            "ingresos",
            "historial_crediticio",
            "deuda",
            "horas_trabajo_semana"
        ]
    ]

    y = df["aprobado_credito"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    evaluar_clasificacion(
        "Regresión logística",
        y_test,
        y_pred
    )


# ==============================
# ÁRBOL DE DECISIÓN
# ==============================

def modelo_arbol_decision(df):

    print("\n================ ÁRBOL DE DECISIÓN ================")

    X = df[
        [
            "edad",
            "ingresos",
            "historial_crediticio",
            "deuda",
            "horas_trabajo_semana"
        ]
    ]

    y = df["aprobado_credito"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    arbol = DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )

    arbol.fit(X_train, y_train)

    y_pred = arbol.predict(X_test)

    evaluar_clasificacion(
        "Árbol de decisión",
        y_test,
        y_pred
    )

    plt.figure(figsize=(16, 8))

    plot_tree(
        arbol,
        feature_names=X.columns,
        class_names=["No aprobado", "Aprobado"],
        filled=True,
        rounded=True
    )

    plt.title("Árbol de decisión")

    plt.savefig(
        OUTPUT_DIR / "arbol_decision.png",
        bbox_inches="tight"
    )

    plt.close()


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================

def main():

    df = cargar_datos()

    modelo_regresion_lineal(df)

    modelo_regresion_logistica(df)

    modelo_arbol_decision(df)

    print("\nProceso finalizado.")
    print("Revise la carpeta outputs/")


if __name__ == "__main__":
    main()

