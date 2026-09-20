import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
from typing import Final

#CAMBIAR ESTO POR UNA BASE DE DATOS EN EL FUTURO (?)
NOMBRE_FICHERO: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "PartidosLigaFutbol20242025.csv"
)

def importar_datos() -> pd.DataFrame:
    """Importamos los datos del CSV a un DataFrame"""
    df: pd.DataFrame = pd.read_csv(
        NOMBRE_FICHERO,
        sep=";",
        parse_dates=["Fecha del partido"],
        dayfirst=True
    )
    return df

def signo(row: pd.Series) -> str:
    """Generamos el símbolo de la quiniela"""
    if row["Goles local"] > row["Goles visitante"]:
        return "1"
    elif row["Goles local"] < row["Goles visitante"]:
        return "2"
    else:
        return "X"

def estadisticas_previas(
    df: pd.DataFrame, 
    equipo: str, 
    fecha: datetime, 
    es_local: bool, 
    ultimos_n: int = 5
) -> tuple [float, float]:
    df_subset: pd.DataFrame = df[df["Fecha del partido"] < fecha]

    if es_local:
        partidos = df_subset[df_subset["Equipo local"] == equipo]
    else:
        partidos = df_subset[df_subset["Equipo visitante"] == equipo]

    if len(partidos) == 0:
        return (0.0, 0.0)

    partidos_recientes = partidos.tail(ultimos_n)

    if es_local:
        media_gf = partidos_recientes["Goles local"].mean()
        media_gc = partidos_recientes["Goles visitante"].mean()
    else:
        media_gf = partidos_recientes["Goles visitante"].mean()
        media_gc = partidos_recientes["Goles local"].mean()

    return (float(media_gf), float(media_gc))

def definir_equipos(df: pd.DataFrame) -> set[str]:
    """Devuelve el conjunto de equipos existentes en el conjunto de datos para mayor exactitud al introducir los nombres"""
    eq_local = df["Equipo local"].to_list()
    eq_visitante = df["Equipo visitante"].to_list()

    return set(eq_local + eq_visitante)

def extraer_features(partidos_a_procesar: pd.DataFrame, df_completo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extrae del dataframe original las estadísticas necesarias según el número de partidos que se requieran"""
    X = list()
    y = list()

    for _, row in partidos_a_procesar.iterrows():
        fecha: datetime = row["Fecha del partido"]
            
        gf_local, gc_local = estadisticas_previas(df=df_completo, equipo=row["Equipo local"], fecha=fecha, es_local=True)
        gf_vis, gc_vis = estadisticas_previas(df=df_completo, equipo=row["Equipo visitante"], fecha= fecha, es_local=False)
            
        if gf_local == 0 and gf_vis == 0:
            continue
            
        X.append([gf_local, gc_local, gf_vis, gc_vis])
        y.append(row["Signo"])

    X_df: pd.DataFrame = pd.DataFrame(
            X,
            columns=["GF_local", "GC_local", "GF_visitante", "GC_visitante"]
        )

    return X_df, y

def baseline(df: pd.DataFrame) -> list[float]:
    """Calcula los porcentajes de victorias para los locales y los visitantes y los empates (baseline)"""
    local_win: int = 0
    visitante_win: int = 0
    empate: int = 0

    for _, row in df.iterrows():
        if row["Goles local"] > row["Goles visitante"]:
            local_win += 1
        elif row["Goles local"] < row["Goles visitante"]:
            visitante_win += 1
        else:
            empate += 1

    num_filas: int = df.shape[0]
    return [local_win*100 / num_filas, visitante_win*100 / num_filas, empate*100 / num_filas]

def walk_forward(df: pd.DataFrame, part_calentamiento: int, modelo: GaussianNB) -> list[float]:
    aciertos_modelo: int = 0
    aciertos_baseline: int = 0
    total_evaluados: int = 0

    X, y = extraer_features(df, df)
    
    for i in range(part_calentamiento, len(X)):
        X_entren, y_entren = X.iloc[:i], y[:i]
        X_test, y_test = X.iloc[i:i+1], y[i]

        modelo.fit(X_entren, y_entren)
        pred = modelo.predict(X_test)[0]

        if pred == y_test:
            aciertos_modelo += 1
        if y_test == "1":
            aciertos_baseline += 1

        total_evaluados += 1

    accuracy_modelo = (aciertos_modelo / total_evaluados) * 100
    accuracy_baseline = (aciertos_baseline / total_evaluados) * 100

    return [accuracy_modelo, accuracy_baseline]

def main() -> None:
    df: pd.DataFrame = importar_datos()
    df["Signo"] = df.apply(signo, axis=1)
    df = df.sort_values("Fecha del partido")

    X_total, y_total = extraer_features(df, df)

    #Usado para poder tratar con los signos de la quiniela "1, X, 2"
    le = LabelEncoder()
    y_enc = le.fit_transform(y_total)

    modelo: GaussianNB = GaussianNB()

    #=====DEBUG=====
    accuracys: list[float] = walk_forward(df, 50, modelo)
    print(f"Accuracy modelo: {accuracys[0]}\nAccuracy baseline {accuracys[1]}")
    #=====DEBUG=====

    modelo.fit(X_total, y_enc)

    equipos: set[str] = definir_equipos(df)
    print(f"Equipos a elegir: {equipos}")

    equipo_local: str = input("Introduce el equipo local: ").strip()
    equipo_visitante: str = input("Introduce el equipo visitante: ").strip()

    if equipo_local not in equipos or equipo_visitante not in equipos or equipo_local == equipo_visitante:
        print("¡ERROR!: Equipos seleccionados no válidos")
        return

    fecha: str = input("Introduce la fecha del partido (DD/MM/AAAA): ")

    try:
        fecha_partido: datetime = datetime.strptime(fecha, "%d/%m/%Y")
    except ValueError:
        print("¡ERROR!: Fecha no válida")
        return

    #=====DEBUG=====
    stats_base: list[float] = baseline(df)
    print(f"% victorias local: {stats_base[0]}\n% victorias visitante: {stats_base[1]}\n% empates {stats_base[2]}")
    #=====DEBUG=====

    gf_l, gc_l = estadisticas_previas(df=df, equipo=equipo_local, fecha=fecha_partido, es_local=True)
    gf_v, gc_v = estadisticas_previas(df=df, equipo=equipo_visitante, fecha=fecha_partido, es_local=False)

    if (gf_l == 0 and gc_l == 0) or (gf_v == 0 and gc_v == 0):
        print("¡ERROR!: No hay suficientes datos históricos para uno de los equipos.")
        return

    X_pred_df = pd.DataFrame(
                    [[gf_l, gc_l, gf_v, gc_v]],
                    columns=["GF_local", "GC_local", "GF_visitante", "GC_visitante"]
                )   

    pred = modelo.predict(X_pred_df)
    resultado: str = le.inverse_transform(pred)[0]

    if resultado == "1":
        print(f"\nPredicción '{equipo_local} vs {equipo_visitante}': {resultado[0]}, gana {equipo_local}")
    elif resultado == "X":
        print(f"\nPredicción '{equipo_local} vs {equipo_visitante}': {resultado[0]}, empate")
    elif resultado == "2":
        print(f"\nPredicción '{equipo_local} vs {equipo_visitante}': {resultado[0]}, gana {equipo_visitante}")
    else:
        print("Error en la predicción")

if __name__ == '__main__':
    main()