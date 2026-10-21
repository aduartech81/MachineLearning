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
        html.P("En este trabajo evalué la clasificación de presencia o ausencia de caries registrada a partir de "
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
        ], open=False),
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
    return bloque("3. Desarrollo y evaluación del modelo propio",
        html.P("Desarrollé un modelo que suma funciones univariadas spline y hasta cinco "
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
            html.Strong("Aporte de este trabajo"),
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
    return bloque("6. Conclusiones, alcance y trabajo futuro",
        html.H3("Respuesta a la pregunta de investigación"),
        html.P(
            "Los predictores de chequeo general aportaron capacidad discriminativa "
            "limitada para clasificar caries registrada. El modelo propio obtuvo "
            "un AUC medio de 0,6051 en la evaluación anidada agrupada, frente a "
            "aproximadamente 0,6057 de regresión logística y gradient boosting."
        ),
        html.H3("Resultado de la contribución implementada"),
        html.P(
            "Se implementó y evaluó un modelo que combina funciones spline, "
            "selección de interacciones y una pérdida mixta de clasificación "
            "y ordenación. Las seis comparaciones pareadas exploratorias no "
            "mostraron evidencia de mejora: todos los intervalos de diferencias "
            "incluyeron cero. Este resultado no demuestra equivalencia."
        ),
        html.H3("Qué muestran las ablaciones y las interacciones"),
        html.P(
            "La variante aditiva obtuvo el mayor AUC medio observado (0,6062), "
            "sin que esto establezca superioridad confirmada. Añadir interacciones "
            "o la pérdida de ranking no produjo una ventaja demostrada en esta "
            "evaluación. Los cinco ajustes del modelo propio convergieron; "
            "edad–ALT y edad–HDL se seleccionaron en cuatro de cinco folds. "
            "Esta repetición parcial no demuestra causalidad ni utilidad predictiva."
        ),
        html.H3("Implicaciones para el uso potencial"),
        html.P(
            "Con el umbral descriptivo de 0,5, la sensibilidad fue prácticamente "
            "nula. Por tanto, estos resultados no respaldan utilizar el modelo "
            "para priorizar atención odontológica. Un umbral operativo tendría "
            "que definirse mediante criterios explícitos y validación interna, "
            "seguida de una evaluación independiente."
        ),
        html.H3("Alcance y limitaciones"),
        html.P(
            "El objetivo es caries registrada, no aparición futura. Agrupar "
            "perfiles idénticos reduce la fuga por duplicados, pero no acredita "
            "independencia por paciente. Los intervalos bootstrap son "
            "condicionales a predicciones guardadas y no incorporan toda la "
            "variabilidad del entrenamiento."
        ),
        html.P(
            "Los componentes del modelo tienen antecedentes. La implementación "
            "propia constituye un desarrollo evaluado, pero la novedad científica "
            "de su combinación no está establecida. La definición diagnóstica, "
            "la procedencia clínica y los metadatos originales requieren "
            "verificación adicional."
        ),
        html.H3("Trabajo futuro"),
        html.P(
            "Ampliar la revisión de antecedentes, verificar la disponibilidad "
            "temporal de las variables y evaluar el enfoque en datos nuevos e "
            "independientes. Las modificaciones motivadas por estos resultados "
            "deberán identificarse como exploratorias; el test histórico ya "
            "consultado no constituye una nueva validación independiente."
        ),
        html.A(
            "Código y evidencias del proyecto independiente",
            href="https://github.com/aduartech81/MachineLearning/tree/"
                 "proyecto-final-independiente-20261018-213747/proyecto_final_caries",
            target="_blank", rel="noopener noreferrer"
        )
    )


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


from proyecto_final_caries.presentacion.explorador_resultados import integrar as integrar_resultados
_revision_callbacks = integrar_resultados(app, SECCIONES, DATA)


from proyecto_final_caries.presentacion.explorador_datos import integrar as integrar_datos
app.config.suppress_callback_exceptions = True
_eda_callback = integrar_datos(app, SECCIONES, DATA)



# Portada del proyecto independiente.
app.layout.children[0] = html.Header([
    html.Div([
        html.Div([
            html.P("ANDRÉS DUARTE CHAVES · MACHINE LEARNING",
                   className="pf-kicker"),
            html.H1("Caries dental: de los datos a la evidencia."),
            html.P(
                "¿Qué pueden aportar los datos de un chequeo general "
                "a la clasificación de caries registrada?",
                className="portada-pregunta"
            ),
            html.P(
                "Exploración de datos, implementación de un modelo propio "
                "y comparación reproducible con modelos de referencia."
            ),
            html.Div([
                html.Span("Clasificación binaria"),
                html.Span("Datos tabulares"),
                html.Span("Evaluación agrupada"),
            ], className="portada-etiquetas")
        ]),
        html.Figure([
            html.Img(
                src=app.get_asset_url("portada_caries.png"),
                alt="Escena ilustrativa de una evaluación odontológica."
            ),
            html.Figcaption(
                "Ilustración generada con IA. No corresponde a participantes "
                "del estudio; el modelo no analiza imágenes."
            )
        ])
    ], className="portada-columnas"),
    html.Div([
        html.Div([html.Strong("55.692"), html.Span("Registros del archivo original")]),
        html.Div([html.Strong("44.582"), html.Span("Filas de desarrollo")]),
        html.Div([html.Strong("23"), html.Span("Predictores utilizados")]),
        html.Div([html.Strong("7"), html.Span("Variantes evaluadas, incluidas ablaciones")]),
    ], className="pf-metrics"),
    html.P(
        "Estudio de caries registrada, no de aparición futura. "
        "El desempeño observado no acredita utilidad clínica.",
        className="portada-alcance"
    )
], className="pf-hero portada-final")


from proyecto_final_caries.presentacion.interpretacion_modelo import integrar as integrar_interpretacion
integrar_interpretacion(SECCIONES, DATA)



# Integración de evidencias V2.
from proyecto_final_caries.presentacion.revision_v2 import integrar as integrar_v2
_v2_callbacks = integrar_v2(app, SECCIONES, DATA)

# Documentación independiente del proyecto final.
@server.route("/informe-final/")
def informe_final():
    return send_from_directory(ROOT / "documentacion", "informe.html")


@server.route("/exposicion/")
def exposicion_final():
    return send_from_directory(ROOT / "documentacion", "exposicion.html")


app.layout.children[0].children.append(html.Div([
    html.A("Informe completo", href="/informe-final/"),
    html.Span(" · "),
    html.A("Guion de exposición · 8 minutos", href="/exposicion/"),
], className="pf-links"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
