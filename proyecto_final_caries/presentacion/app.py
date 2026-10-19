"""Aplicación independiente del proyecto final; no entrena modelos."""
from pathlib import Path
import json
import os
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from flask import send_from_directory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "evidencias"
app = Dash(__name__, assets_folder=str(ROOT / "assets"))
server = app.server
app.title = "Proyecto final · Caries dental"


def tabla(nombre):
    df = pd.read_csv(DATA / nombre).fillna("")
    def mostrar(v):
        return f"{v:.5f}" if isinstance(v, float) else str(v)
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(c) for c in df.columns])),
        html.Tbody([
            html.Tr([html.Td(mostrar(v)) for v in fila])
            for fila in df.itertuples(index=False, name=None)
        ])
    ]), className="pf-table")


def descarga(nombre, titulo):
    return html.A(titulo, href="/evidencias/" + nombre, className="pf-button")


def figura(nombre, texto):
    return html.Figure([
        html.Img(src="/evidencias/" + nombre, alt=texto),
        html.Figcaption(texto)
    ])


def bloque(titulo, *contenido):
    return html.Section([html.H2(titulo), *contenido], className="pf-section")


def problema():
    return bloque("1. Problema y utilidad potencial",
        html.P("Clasificar presencia o ausencia de caries registrada a partir de "
               "datos de chequeo general. El diseño transversal no permite afirmar "
               "que se predice la aparición futura de caries."),
        html.P("Se utiliza smoking.csv: 55.692 registros. El desarrollo comprende "
               "44.582 filas y 35.641 perfiles; el test histórico contiene 11.110 filas "
               "y no interviene en la evaluación anidada presentada."),
        html.P("Uso potencial: apoyar la priorización de evaluación odontológica "
               "cuando los predictores ya estén disponibles. La secuencia temporal "
               "de recogida no está acreditada y el rendimiento no respalda uso clínico."),
        html.P("La fuente de distribución identificada es Body Signal of Smoking. "
               "Permanecen pendientes verificaciones de licencia, protocolo original "
               "y definición diagnóstica."),
        html.A("Consultar fuente de distribución",
               href="https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking",
               target="_blank", rel="noopener noreferrer"))


def datos():
    return bloque("2. Datos, variables y exploración",
        html.P("Se conservan 23 predictores y se excluyen ID, oral y tartar. "
               "Las copias de un perfil permanecen juntas. Esto no garantiza "
               "independencia entre pacientes."),
        html.P("Las tablas exploratorias proceden del entrenamiento. Las categorías "
               "no se tratan como magnitudes. Las unidades y códigos no confirmados "
               "se identifican explícitamente en el diccionario."),
        html.Details([
            html.Summary("Diccionario: significado, tipo y tratamiento"),
            tabla("diccionario.csv")
        ], open=True),
        html.Details([html.Summary("Descriptivos numéricos"), tabla("descriptive.csv")]),
        html.Details([
            html.Summary("Exploración categórica · perfiles únicos de entrenamiento"),
            tabla("categorical_eda.csv")
        ]),
        html.P("Los extremos se conservan. Los rangos observados no son intervalos "
               "clínicos de referencia ni intervalos de confianza."),
        figura("curvas_aprendizaje.png",
               "Curvas del modelo logístico base: ajuste y validación interna; "
               "no representan la evolución del modelo propio."),
        descarga("diccionario.csv", "Descargar diccionario"))


def modelo():
    antecedentes = json.loads((DATA / "antecedentes.json").read_text())
    return bloque("3. Aporte candidato e implementación",
        html.P("El modelo suma funciones univariadas spline y hasta cinco "
               "interacciones numéricas. Combina pérdida logística de clasificación, "
               "pérdida de ordenación positivo–negativo y regularización L2."),
        dcc.Markdown(
            r"$$L=(1-\alpha)L_{\mathrm{BCE}}"
            r"+\alpha\,\operatorname{media}\log(1+\exp(-(s_+-s_-)))"
            r"+\frac{\lambda}{2}\Vert\beta\Vert_2^2$$",
            mathjax=True
        ),
        html.P("Las interacciones se seleccionan por estabilidad exploratoria de "
               "asociaciones con residuos en submuestras de perfiles. El intercepto "
               "no se penaliza. Los gradientes se comprobaron numéricamente."),
        html.Div([
            html.Strong("Estado del aporte"),
            html.P("Los componentes tienen antecedentes. La implementación propia "
                   "no demuestra novedad científica de la combinación. Las ablaciones "
                   "no mostraron evidencia de ventaja de los componentes añadidos.")
        ], className="pf-finding"),
        html.H3("Antecedentes iniciales"),
        html.Ul([
            html.Li([
                html.A(a["trabajo"], href=a["url"], target="_blank",
                       rel="noopener noreferrer"),
                html.P(a["implicacion"])
            ]) for a in antecedentes
        ]),
        descarga("especificacion_modelo.md", "Especificación completa"),
        descarga("plan_comparacion_v1.json", "Plan de comparación"))


def evaluacion():
    return bloque("4. Diseño experimental e incertidumbre",
        html.P("Validación cruzada anidada agrupada: cinco folds externos y tres "
               "internos. Cada familia tiene cuatro candidatos; se selecciona por "
               "AUC medio interno. Preprocesamiento y selección ocurren dentro del ajuste."),
        html.P("Referencias: regresión logística, SVM lineal y gradient boosting. "
               "Ablaciones: sin interacciones, sin ranking y modelo aditivo. "
               "Todos comparten las particiones. Dummy tiene AUC 0,5 dentro de cada fold."),
        html.P("Los intervalos individuales del 95 % proceden de 2.000 réplicas bootstrap "
               "por perfiles, dentro de fold y clase. El estadístico es el AUC medio "
               "entre folds. Los seis contrastes exploratorios usan corrección Holm."),
        html.P("La inferencia está condicionada a predicciones fijas: no incluye toda "
               "la variabilidad de reentrenamiento ni la dependencia entre modelos "
               "ajustados sobre conjuntos solapados. No es validación externa."),
        figura("auc_intervalos.png", "AUC medio e intervalos condicionales del 95 %."),
        descarga("metodo_inferencia.json", "Método inferencial"),
        descarga("resultados_por_fold.csv", "Resultados y parámetros por fold"))


def resultados():
    return bloque("5. Resultados, calibración y errores",
        tabla("resumen_modelos.csv"),
        html.P("Umbrales descriptivos: probabilidad 0,5 y puntuación 0 para SVM. "
               "No se seleccionaron para maximizar resultados externos. La sensibilidad "
               "es prácticamente nula: una alta especificidad no compensa esa limitación."),
        figura("roc_pr.png",
               "ROC y precisión–recall de predicciones OOF reunidas. "
               "Su AUC no equivale necesariamente al promedio de AUC entre folds."),
        figura("calibracion_matriz.png",
               "Calibración por grupos y matriz de confusión del modelo propio."),
        html.P("Los falsos negativos son casos registrados no señalados por el modelo. "
               "Los falsos positivos podrían ocasionar evaluaciones adicionales. "
               "Un umbral operativo requiere criterios explícitos y selección interna."),
        html.H3("Diferencias pareadas"),
        tabla("comparaciones.csv"),
        descarga("resumen_modelos.csv", "Descargar métricas"),
        descarga("comparaciones.csv", "Descargar contrastes"))


def conclusiones():
    return bloque("6. Conclusiones y defensa",
        html.P("El modelo propio obtuvo AUC medio 0,6051. No se encontró evidencia "
               "de mejora frente a las referencias ni de beneficio de sus componentes "
               "añadidos. Todos los intervalos de diferencias incluyen cero; "
               "esto no demuestra equivalencia."),
        html.P("La versión aditiva tiene el mayor promedio observado, pero no se "
               "declara ganadora confirmada mediante selección retrospectiva."),
        html.H3("Preguntas para la sustentación"),
        html.Ul([
            html.Li("¿Por qué clasificación? El objetivo registra dos estados de caries."),
            html.Li("¿Por qué esa codificación? Los códigos categóricos no implican distancias numéricas."),
            html.Li("¿Qué significa el intervalo? Incertidumbre condicional de evaluación, no predicción individual."),
            html.Li("¿Qué aportamos? Una implementación evaluada mediante ablaciones; su novedad científica sigue pendiente."),
            html.Li("¿Es clínicamente utilizable? No está validado para ese uso.")
        ]),
        html.P("Pendientes: profundizar antecedentes, verificar metadatos originales, "
               "completar interpretación específica del estimador y obtener evaluación "
               "independiente. No se garantiza una calificación."),
        html.A("Repositorio y trazabilidad",
               href="https://github.com/aduartech81/MachineLearning/tree/main/proyecto_final_caries",
               target="_blank", rel="noopener noreferrer"))


SECCIONES = {
    "problema": problema, "datos": datos, "modelo": modelo,
    "evaluacion": evaluacion, "resultados": resultados, "conclusiones": conclusiones
}
app.layout = html.Div([
    html.Header([
        html.P("ANDRÉS DUARTE CHAVES · MACHINE LEARNING", className="pf-kicker"),
        html.H1("Proyecto final: caries dental"),
        html.P("Datos, modelo implementado y evaluación reproducible."),
        html.Span("Avance académico · sin utilidad clínica acreditada", className="pf-badge")
    ], className="pf-hero"),
    dcc.Tabs(id="final-tabs", value="problema", children=[
        dcc.Tab(label=titulo, value=clave) for clave, titulo in [
            ("problema", "1 · Problema"), ("datos", "2 · Datos"),
            ("modelo", "3 · Aporte"), ("evaluacion", "4 · Evaluación"),
            ("resultados", "5 · Resultados"), ("conclusiones", "6 · Conclusiones")
        ]
    ]),
    html.Main(id="final-content", className="pf-main"),
    html.Footer("Maestría en Ingeniería Biomédica · Docente: Dr. Lihki Rubio",
                className="pf-footer")
], className="pf-shell")


@app.callback(Output("final-content", "children"), Input("final-tabs", "value"))
def mostrar(clave):
    return SECCIONES.get(clave, problema)()


@server.route("/evidencias/<nombre>")
def evidencia(nombre):
    permitidos = {p.name for p in DATA.iterdir() if p.is_file()}
    if nombre not in permitidos:
        return "Archivo no disponible", 404
    return send_from_directory(str(DATA), nombre,
                               as_attachment=not nombre.endswith(".png"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
