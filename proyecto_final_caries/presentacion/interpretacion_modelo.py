
"""Interpretación descriptiva de interacciones de modelos guardados."""
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc

def integrar(secciones, data):
    detalle = pd.read_csv(data / "interacciones_por_fold.csv")
    resumen = pd.read_csv(data / "interacciones_resumen.csv")
    folds = sorted(detalle["fold"].unique())
    pares = resumen["interaccion"].tolist()
    matriz = (
        detalle.assign(presente=1)
        .pivot(index="interaccion", columns="fold", values="presente")
        .reindex(index=pares, columns=folds)
        .fillna(0)
    )
    fig = go.Figure(go.Heatmap(
        z=matriz.values,
        x=[f"Fold {i+1}" for i in range(len(folds))],
        y=pares,
        zmin=0, zmax=1,
        colorscale=[[0, "#edf3f5"], [1, "#147d82"]],
        showscale=False,
        text=matriz.replace({0: "No seleccionada", 1: "Seleccionada"}).values,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_white", height=max(450, 32 * len(pares)),
        margin=dict(l=200, r=25, t=35, b=45),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Arial", color="#243b46")
    )
    contraste_v2 = pd.read_csv(data / "revision_v2_20261020_154421_004423" / "contrastes/comparaciones_pareadas.csv")
    c = contraste_v2.loc[contraste_v2.comparacion == "Propio - Sin_interacciones"].iloc[0]
    intervalo_v2 = (f"{c.ic95_inferior_condicional:.6f} a {c.ic95_superior_condicional:.6f}").replace(".", ",")
    anterior = secciones["modelo"]

    def modelo_ampliado():
        contenido = anterior()
        filas = [
            html.Tr([
                html.Td(r.interaccion),
                html.Td(f"{r.folds_seleccionada}/5"),
                html.Td(f"{r.frecuencia_interna_media_al_seleccionarse:.2f}")
            ])
            for r in resumen.itertuples(index=False)
        ]
        contenido.children.extend([
            html.H3("Interacciones seleccionadas en los cinco ajustes externos"),
            html.P(
                "Los cinco ajustes del modelo propio registraron convergencia. "
                "Cada uno seleccionó cinco interacciones. La matriz permite "
                "examinar cuáles se repiten cuando cambia el entrenamiento."
            ),
            dcc.Graph(figure=fig, config={"displaylogo": False}),
            html.Div([
                html.Strong("Hallazgo principal"),
                html.P(
                    "Edad–HDL y edad–ALT se seleccionaron en cuatro de cinco "
                    "folds. Ninguna interacción se seleccionó en los cinco. "
                    "Existe repetición parcial, con variación en la composición "
                    "de las interacciones entre ajustes."
                )
            ], className="pf-finding"),
            html.Div(html.Table([
                html.Thead(html.Tr([
                    html.Th("Interacción"),
                    html.Th("Folds con selección"),
                    html.Th("Frecuencia interna media, solo cuando fue seleccionada")
                ])),
                html.Tbody(filas)
            ]), className="pf-table"),
            html.H3("Qué significa esta selección"),
            html.P(
                "La frecuencia interna procede de las submuestras de perfiles "
                "utilizadas dentro de cada ajuste. El recuento entre folds es "
                "otra medida: describe repetición en cinco entrenamientos "
                "solapados. No equivale a cinco confirmaciones independientes."
            ),
            html.P(
                "La asociación con residuos fue un criterio exploratorio de "
                "selección. No es el coeficiente final de la interacción, no "
                "indica por sí sola la dirección de su efecto y no demuestra "
                "un mecanismo biológico."
            ),
            html.H3("¿Las interacciones mejoraron la clasificación?"),
            html.P(
                "La comparación pareada del modelo propio con la variante sin "
                "interacciones dio una diferencia de AUC medio aproximada de "
                f"−0,000030, con IC condicional del 95 % de {intervalo_v2} (análisis V2). "
                "No se encontró evidencia de mejora; tampoco se demostró "
                "equivalencia. La repetición de una interacción no implica "
                "que aporte una ganancia predictiva."
            ),
            html.A("Descargar selección por fold",
                   href="/evidencias/interacciones_por_fold.csv",
                   className="pf-button"),
            html.A("Descargar resumen de repetición",
                   href="/evidencias/interacciones_resumen.csv",
                   className="pf-button"),
        ])
        return contenido
    secciones["modelo"] = modelo_ampliado
