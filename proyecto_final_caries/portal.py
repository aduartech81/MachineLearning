"""Portada del proyecto final; sólo presenta evidencias guardadas."""
from pathlib import Path
import pandas as pd
from dash import Input, Output, dcc, html
from flask import send_from_directory

ROOT = Path(__file__).resolve().parents[1]
EVIDENCIAS = ROOT / "_static/proyecto_final_caries"
BOOK = "https://aduartech81.github.io/MachineLearning/"
REPO = "https://github.com/aduartech81/MachineLearning"
PERMITIDOS = {
    "roc_pr.png", "calibracion_matriz.png", "auc_intervalos.png",
    "resumen_modelos.csv", "resultados_por_fold.csv", "comparaciones.csv",
    "auc_ic95.csv", "metodo_inferencia.json", "antecedentes.json",
    "especificacion_modelo.md", "plan_comparacion_v1.json"
}


def enlace(texto, url, clase="pf-button"):
    return html.A(texto, href=url, className=clase)


def figura(nombre, descripcion):
    return html.Figure([
        html.Img(src="/evidencias-final/" + nombre, alt=descripcion),
        html.Figcaption(descripcion)
    ])


def seccion(numero, titulo, contenido, identificador):
    return html.Section([
        html.P(f"SECCIÓN {numero}", className="pf-kicker"),
        html.H2(titulo), *contenido
    ], id=identificador, className="pf-section")


def portada():
    tabla = pd.read_csv(EVIDENCIAS / "resumen_modelos.csv")
    propio = tabla.set_index("modelo").loc["Propio"]
    filas = []
    for _, r in tabla.iterrows():
        filas.append(html.Tr([
            html.Td(r["modelo"]),
            html.Td(f"{r['auc_medio_folds']:.4f}"),
            html.Td(f"{r['sensibilidad']:.2%}"),
            html.Td(f"{r['especificidad']:.2%}"),
            html.Td("No disponible" if pd.isna(r["brier"]) else f"{r['brier']:.4f}")
        ]))

    return html.Div([
        html.Header([
            html.P("MACHINE LEARNING · ANDRÉS DUARTE CHAVES", className="pf-kicker"),
            html.H1("Caries dental"),
            html.P("De los datos de chequeo a una evaluación reproducible de modelos."),
            html.Span("Avance del proyecto final · evaluación interna", className="pf-badge"),
            html.Div([
                enlace("Leer informe completo", BOOK + "proyecto_final_caries.html"),
                enlace("Explorar Entregable 2", "/entregable-2"),
                enlace("Código y evidencias", REPO)
            ], className="pf-links")
        ], className="pf-hero"),
        html.Nav([
            html.A(t, href="#" + i) for t, i in [
                ("Problema", "problema"), ("Datos", "datos"),
                ("Aporte", "aporte"), ("Evaluación", "evaluacion"),
                ("Resultados", "resultados"), ("Conclusiones", "conclusiones")
            ]
        ], className="pf-nav"),
        html.Main([
            html.Div([
                html.Div([html.Strong("44.582"), html.Span("filas de desarrollo")]),
                html.Div([html.Strong("35.641"), html.Span("perfiles agrupados")]),
                html.Div([html.Strong("23"), html.Span("predictores")]),
                html.Div([html.Strong(f"{propio['auc_medio_folds']:.4f}"),
                          html.Span("AUC medio · modelo propio")])
            ], className="pf-metrics"),
            html.Div([
                html.Strong("Qué encontramos"),
                html.P("La implementación funciona, pero no mostró una mejora demostrada "
                       "frente a las referencias. Su originalidad científica no está acreditada. "
                       "Los resultados no respaldan uso clínico.")
            ], className="pf-finding"),

            seccion(1, "Problema y utilidad potencial", [
                html.P("Clasificar presencia o ausencia de caries registrada en smoking.csv. "
                       "El estudio es transversal: no predice aparición futura."),
                html.P("La priorización de una evaluación odontológica es un uso potencial "
                       "que requeriría validación adicional y disponibilidad previa de los predictores.")
            ], "problema"),

            seccion(2, "Datos, variables y entregables", [
                html.P("Los perfiles repetidos permanecen juntos. Se excluyen ID, oral y tartar. "
                       "La agrupación por perfil no garantiza independencia por paciente."),
                html.P("El diccionario conserva explícitas las unidades, códigos y aspectos "
                       "de procedencia pendientes de confirmación."),
                html.Div([
                    enlace("Entregable 1 · EDA y modelo base",
                           BOOK + "entregable_1_clasificacion.html"),
                    enlace("Entregable 2 · dashboard", "/entregable-2")
                ], className="pf-links")
            ], "datos"),

            seccion(3, "Implementación y aporte candidato", [
                html.P("Modelo aditivo con funciones spline, hasta cinco interacciones "
                       "numéricas y pérdida combinada de clasificación y ranking."),
                html.P("Sus componentes tienen antecedentes. Las ablaciones evalúan las "
                       "versiones sin interacciones, sin ranking y únicamente aditiva."),
                html.Details([
                    html.Summary("Consultar especificación y antecedentes"),
                    enlace("Especificación", "/evidencias-final/especificacion_modelo.md"),
                    enlace("Antecedentes iniciales", "/evidencias-final/antecedentes.json")
                ])
            ], "aporte"),

            seccion(4, "Evaluación e incertidumbre", [
                html.P("Validación cruzada anidada: cinco folds externos y tres internos. "
                       "Cada familia usa cuatro candidatos. Las transformaciones y selección "
                       "se ajustan dentro del entrenamiento correspondiente."),
                html.P("El test histórico ya fue consultado y no se utiliza en este experimento. "
                       "Los intervalos bootstrap por perfiles son condicionales a predicciones "
                       "fijas; no incluyen toda la variabilidad de reentrenamiento. "
                       "Los contrastes son exploratorios, con corrección Holm."),
                figura("auc_intervalos.png",
                       "AUC medio entre folds e intervalos individuales del 95 %; análisis condicional.")
            ], "evaluacion"),

            seccion(5, "Resultados y errores", [
                html.Div(html.Table([
                    html.Thead(html.Tr([html.Th(t) for t in [
                        "Modelo", "AUC medio", "Sensibilidad", "Especificidad", "Brier"
                    ]])),
                    html.Tbody(filas)
                ]), className="pf-table"),
                html.P("Umbrales descriptivos: probabilidad 0,5 y puntuación 0 para SVM. "
                       "No se optimizaron sobre resultados externos. La sensibilidad es "
                       "prácticamente nula con estos cortes."),
                figura("roc_pr.png",
                       "Curvas descriptivas de predicciones OOF reunidas; no equivalen al AUC medio entre folds."),
                figura("calibracion_matriz.png",
                       "Calibración por grupos de probabilidad y errores del modelo propio."),
                enlace("Descargar métricas", "/evidencias-final/resumen_modelos.csv")
            ], "resultados"),

            seccion(6, "Conclusiones y defensa", [
                html.P("Los seis intervalos de diferencias incluyen cero. No se encontró "
                       "evidencia de ventaja del modelo completo; esto no demuestra equivalencia."),
                html.P("El prototipo es un avance académico implementado y evaluado. "
                       "Quedan pendientes la justificación de originalidad, verificaciones "
                       "de procedencia y validación independiente."),
                html.Div([
                    enlace("Informe y respuestas para defensa", BOOK + "proyecto_final_caries.html"),
                    enlace("Contrastes", "/evidencias-final/comparaciones.csv"),
                    enlace("Método estadístico", "/evidencias-final/metodo_inferencia.json")
                ], className="pf-links")
            ], "conclusiones")
        ], className="pf-main"),
        html.Footer("Andrés Duarte Chaves · Maestría en Ingeniería Biomédica · "
                    "Docente: Dr. Lihki Rubio", className="pf-footer")
    ], className="pf-shell")


def integrar(app):
    layout_entregable = app.layout
    validacion_anterior = app.validation_layout

    @app.server.route("/evidencias-final/<nombre>", endpoint="evidencia_final")
    def evidencia_final(nombre):
        if nombre not in PERMITIDOS:
            return "Archivo no disponible", 404
        return send_from_directory(
            str(EVIDENCIAS), nombre,
            as_attachment=not nombre.endswith(".png")
        )

    app.title = "Caries dental | Proyecto y entregables"
    app.layout = html.Div([
        dcc.Location(id="pf-url", refresh=False),
        html.Div(id="pf-contenido")
    ])
    app.validation_layout = html.Div([
        app.layout, validacion_anterior, portada()
    ])

    @app.callback(Output("pf-contenido", "children"), Input("pf-url", "pathname"))
    def mostrar_ruta(pathname):
        if pathname in (None, "/", "/proyecto-final", "/proyecto-final/"):
            return portada()
        if pathname in ("/entregable-2", "/entregable-2/"):
            return html.Div([
                html.Div(enlace("← Volver al proyecto", "/"), className="pf-return"),
                layout_entregable
            ])
        return html.Div([
            html.H2("Página no encontrada"), enlace("Volver al inicio", "/")
        ])
    return mostrar_ruta
