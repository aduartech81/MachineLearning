
"""EDA agregado de entrenamiento; sin entrenamiento de modelos."""
import json
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

def integrar(app, secciones, data):
    evidencia = json.loads((data / "eda_interactivo.json").read_text())
    variables = evidencia["variables"]
    anterior = secciones["datos"]

    def datos_ampliados():
        contenido = anterior()
        contenido.children.insert(1, html.Div([
            html.P("EXPLORACIÓN DEL ENTRENAMIENTO", className="pf-kicker"),
            html.H3("Cómo cambian las distribuciones entre las dos clases"),
            html.P(
                "44.582 filas de entrenamiento. Selecciona cualquiera de los "
                "23 predictores. Los porcentajes se calculan dentro de cada "
                "clase sobre valores observados; las copias de perfiles "
                "conservan su peso por fila."
            ),
            html.Label("Variable", htmlFor="eda-final-variable"),
            dcc.Dropdown(
                id="eda-final-variable",
                options=[{"label": v, "value": v} for v in variables],
                value="age", clearable=False
            ),
            dcc.Graph(id="eda-final-grafico", config={"displaylogo": False}),
            html.Div(id="eda-final-resumen"),
            html.P(
                "En las numéricas se usan 40 intervalos comunes a ambas clases "
                "y se conserva todo el rango. Puedes ampliar el gráfico para "
                "examinar el centro de la distribución. Estos intervalos son "
                "rangos gráficos, no intervalos de confianza ni categorías "
                "utilizadas para entrenar el modelo."
            ),
            html.P(
                "Los códigos categóricos se muestran como están registrados; "
                "consulta su significado y las unidades verificadas en el "
                "diccionario. Las diferencias son descriptivas y no demuestran "
                "causalidad ni independencia entre pacientes."
            ),
            html.A("Descargar estadísticas del explorador",
                   href="/evidencias/eda_interactivo.json",
                   className="pf-button")
        ], className="revision-explorador"))
        return contenido

    secciones["datos"] = datos_ampliados

    @app.callback(
        Output("eda-final-grafico", "figure"),
        Output("eda-final-resumen", "children"),
        Input("eda-final-variable", "value")
    )
    def actualizar(nombre):
        entrada = variables[nombre]
        numerica = entrada["tipo"] == "numerica"
        fig = go.Figure()
        filas = []
        for clase, color in (("0", "#147d82"), ("1", "#bd693d")):
            g = entrada["grupos"][clase]
            etiqueta = evidencia["clases"][clase]
            porcentajes = [
                100 * n / g["observados"] if g["observados"] else 0
                for n in g["conteos"]
            ]
            parametros = {}
            if numerica:
                x = entrada["centros"]
                parametros["width"] = entrada["anchos"]
                custom = [
                    [n, entrada["bordes"][i], entrada["bordes"][i+1]]
                    for i, n in enumerate(g["conteos"])
                ]
                hover = (
                    "Intervalo: %{customdata[1]:.4g} a %{customdata[2]:.4g}"
                    "<br>Filas: %{customdata[0]}<br>Porcentaje: %{y:.2f}%"
                    "<extra>%{fullData.name}</extra>"
                )
            else:
                x = entrada["categorias"]
                custom = [[n] for n in g["conteos"]]
                hover = (
                    "Categoría: %{x}<br>Filas: %{customdata[0]}"
                    "<br>Porcentaje: %{y:.2f}%<extra>%{fullData.name}</extra>"
                )
            fig.add_trace(go.Bar(
                x=x, y=porcentajes, name=etiqueta,
                marker_color=color, opacity=.72 if numerica else 1,
                customdata=custom, hovertemplate=hover, **parametros
            ))
            filas.append(html.Tr([
                html.Td(etiqueta), html.Td(f'{g["observados"]:,}'),
                html.Td(str(g["faltantes"])),
                html.Td(
                    " · ".join(f"{k}: {v:.3g}" for k, v in g.get("resumen", {}).items())
                    if numerica else "Categorías conservadas sin agrupar."
                )
            ]))
        fig.update_layout(
            template="plotly_white", height=450,
            barmode="overlay" if numerica else "group",
            xaxis_title=nombre, yaxis_title="% de valores observados de cada clase",
            legend_title_text="", margin=dict(t=30, b=75, l=55, r=20),
            font=dict(family="Arial", color="#243b46")
        )
        if not numerica:
            fig.update_xaxes(type="category")
        resumen = html.Div(html.Table([
            html.Thead(html.Tr([
                html.Th("Clase"), html.Th("Observados"),
                html.Th("Faltantes"), html.Th("Resumen descriptivo")
            ])), html.Tbody(filas)
        ]), className="pf-table")
        return fig, resumen
    return actualizar
