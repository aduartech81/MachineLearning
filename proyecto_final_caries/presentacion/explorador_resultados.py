
"""Exploración de resultados guardados; no entrena modelos."""
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html, dash_table

ETIQUETAS = {
    "modelo": "Modelo", "family": "Familia", "familia": "Familia",
    "auc_medio_folds": "AUC medio entre folds",
    "auc_oof": "AUC de predicciones OOF reunidas",
    "sd_auc_folds": "Desviación del AUC entre folds",
    "sensibilidad": "Sensibilidad", "especificidad": "Especificidad",
    "precision": "Precisión", "brier": "Puntuación de Brier",
    "segundos_totales": "Tiempo total observado (s)",
}

def integrar(app, secciones, data):
    fuentes = {
        "resumen": ("Siete variantes evaluadas", "resumen_modelos.csv"),
        "folds": ("Resultados de los folds externos", "resultados_por_fold.csv"),
        "contrastes": ("Comparaciones pareadas", "comparaciones.csv"),
        "intervalos": ("Intervalos condicionales del AUC", "auc_ic95.csv"),
    }
    tablas = {
        k: pd.read_csv(data / archivo, float_precision="round_trip")
        for k, (_, archivo) in fuentes.items()
    }

    def tabla_completa(clave):
        df = tablas[clave]
        registros = df.astype(object).where(pd.notna(df), None).to_dict("records")
        return dash_table.DataTable(
            data=registros,
            columns=[
                {"name": ETIQUETAS.get(c, c.replace("_", " ")), "id": c}
                for c in df.columns
            ],
            page_size=10,
            sort_action="native",
            filter_action="native",
            export_format="csv",
            style_table={"overflowX": "auto"},
            style_cell={
                "fontFamily": "Arial, sans-serif", "fontSize": 13,
                "padding": "12px", "textAlign": "left",
                "minWidth": "115px", "maxWidth": "340px",
                "whiteSpace": "normal", "height": "auto",
            },
            style_header={
                "backgroundColor": "#e9f3f2", "fontWeight": "bold",
                "color": "#173d46",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f7fafb"}
            ],
        )

    anterior = secciones["resultados"]

    def resultados_ampliados():
        contenido = anterior()
        explorador = html.Div([
            html.P("EXPLORADOR DE EVIDENCIA", className="pf-kicker"),
            html.H3("Cada resultado, con su contexto"),
            html.P(
                "Selecciona la tabla y la métrica. Las barras representan valores "
                "guardados, sin intervalos añadidos. Consulta los intervalos "
                "condicionales y los contrastes en sus tablas correspondientes."
            ),
            html.Div([
                html.Div([
                    html.Label("Nivel de detalle", htmlFor="revision-fuente"),
                    dcc.Dropdown(
                        id="revision-fuente",
                        options=[
                            {"label": fuentes[k][0], "value": k}
                            for k in ("resumen", "folds")
                        ],
                        value="resumen", clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Métrica", htmlFor="revision-metrica"),
                    dcc.Dropdown(id="revision-metrica", clearable=False),
                ]),
            ], className="revision-controles"),
            dcc.Graph(
                id="revision-grafico",
                config={"displaylogo": False, "toImageButtonOptions": {"scale": 2}},
            ),
            html.P(
                "AUC medio por fold y AUC OOF reunido son estadísticas diferentes. "
                "La sensibilidad y especificidad corresponden a los umbrales "
                "ya documentados; estos controles no seleccionan un nuevo umbral."
            ),
            html.H3("Resultados completos y descargables"),
            *[
                html.Details([
                    html.Summary(fuentes[k][0]),
                    tabla_completa(k),
                ], open=(k == "resumen"))
                for k in fuentes
            ],
        ], className="revision-explorador")
        contenido.children.insert(1, explorador)
        return contenido

    secciones["resultados"] = resultados_ampliados

    @app.callback(
        Output("revision-metrica", "options"),
        Output("revision-metrica", "value"),
        Input("revision-fuente", "value"),
    )
    def opciones(clave):
        df = tablas[clave]
        columnas = list(df.select_dtypes(include="number").columns)
        opciones = [
            {"label": ETIQUETAS.get(c, c.replace("_", " ")), "value": c}
            for c in columnas
        ]
        preferida = next(
            (c for c in columnas if "auc" in c.lower()
             and "sd" not in c.lower() and "std" not in c.lower()),
            columnas[0] if columnas else None,
        )
        return opciones, preferida

    @app.callback(
        Output("revision-grafico", "figure"),
        Input("revision-fuente", "value"),
        Input("revision-metrica", "value"),
    )
    def grafico(clave, metrica):
        df = tablas[clave].copy()
        if metrica not in df.select_dtypes(include="number").columns:
            return px.scatter(title="Selecciona una métrica disponible.")
        modelo = next(
            (c for c in ("modelo", "familia", "family", "model")
             if c in df.columns), None
        )
        df["registro"] = [f"Registro {i+1}" for i in range(len(df))]
        eje = modelo if clave == "resumen" and modelo else "registro"
        fig = px.bar(
            df, x=eje, y=metrica, color=modelo,
            hover_data=list(df.columns),
            labels={**ETIQUETAS, "registro": "Registro de evaluación"},
            color_discrete_sequence=[
                "#147d82", "#344e73", "#d18a3b", "#954b69",
                "#5b8370", "#8671a1", "#698794",
            ],
        )
        fig.update_layout(
            template="plotly_white", height=460,
            margin=dict(l=35, r=25, t=35, b=90),
            font=dict(family="Arial", color="#243b46"),
            legend_title_text="Modelo",
        )
        return fig

    return opciones, grafico
