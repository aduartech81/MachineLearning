
"""Presentación V2: solo evidencias agregadas; sin entrenamiento."""
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from flask import send_from_directory

PAQUETE = "revision_v2_20261020_154421_004423"

def integrar(app, secciones, data):
    base = Path(data) / PAQUETE
    manifiesto = json.loads((base / "manifest.json").read_text())
    permitidos = {r["archivo"] for r in manifiesto["archivos"]}
    permitidos |= {"LEEME.md", "manifest.json"}
    anteriores = {k: secciones[k] for k in
                  ("modelo", "evaluacion", "resultados", "conclusiones")}

    def leer(nombre):
        return pd.read_csv(base / nombre)

    def url(nombre):
        return "/evidencias-v2/" + nombre

    def enlace(nombre, texto):
        return html.A(texto, href=url(nombre), className="pf-button",
                      target="_blank", rel="noopener noreferrer")

    def imagen(nombre, texto):
        return html.Figure([
            html.Img(src=url(nombre), alt=texto,
                     style={"width": "100%", "height": "auto"}),
            html.Figcaption(texto)
        ])

    etiquetas = {
        "familia": "Modelo", "modelo": "Modelo", "folds": "Folds",
        "auc_medio": "AUC medio", "auc_medio_folds": "AUC medio",
        "sd_auc_folds": "DE entre folds", "sensibilidad_media": "Sensibilidad",
        "especificidad_media": "Especificidad", "precision_media": "VPP",
        "f1_medio": "F1", "brier_medio": "Brier", "comparacion": "Comparación",
        "delta_auc_medio": "Diferencia de AUC",
        "ic95_inferior_condicional": "IC 95 % inferior",
        "ic95_superior_condicional": "IC 95 % superior",
        "p_bootstrap_centrado_exploratorio": "p exploratorio",
        "p_holm_10_exploratorio": "p ajustado (Holm)",
        "optimizador": "Búsqueda", "balanceo": "Balanceo",
        "tipo": "Caso", "row_id": "ID", "probabilidad_calibrada": "Probabilidad calibrada",
        "r2_local_min": "R² local mínimo", "r2_local_max": "R² local máximo",
        "error_absoluto_medio": "Error absoluto medio", "error_absoluto_max": "Error absoluto máximo",
        "variante": "Umbral", "filas": "Filas", "auc_oof_agrupado": "AUC OOF",
        "average_precision_oof": "Precisión promedio (AP)", "brier_oof": "Brier",
        "log_loss_oof": "Pérdida logarítmica", "accuracy": "Exactitud",
        "sensibilidad": "Sensibilidad", "especificidad": "Especificidad",
        "precision": "VPP", "f1": "F1", "tn": "VN", "fp": "FP",
        "fn": "FN", "tp": "VP", "ece_10_bins_fijos": "ECE (10 intervalos)"
    }
    nombres = {
        "Propio": "Modelo propio", "Logistica_L1_L2": "Regresión logística",
        "SVM": "SVM lineal", "RandomForest": "Bosque aleatorio",
        "DecisionTree": "Árbol de decisión", "GaussianNB": "Naive Bayes gaussiano",
        "Sin_interacciones": "Sin interacciones", "Sin_ranking": "Sin ranking",
        "Aditivo": "Aditivo", "sin_balanceo": "Sin balanceo", "pesos": "Ponderación",
        "grid": "Cuadrícula", "random": "Aleatoria", "bayes": "TPE",
        "genetica": "Genética", "smote": "SMOTE", "adasyn": "ADASYN",
        "Sigmoide_0.5": "0,5", "Sigmoide_Youden_interno": "Youden interno"
    }

    def tabla(df):
        def valor(v, columna):
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                decimales = 6 if columna.startswith(("delta_", "ic95_", "p_")) else 4
                return f"{v:.{decimales}f}".replace(".", ",")
            return nombres.get(str(v), str(v)).replace("Propio - ", "Modelo propio − ").replace("_", " ")
        return html.Div(html.Table([
            html.Thead(html.Tr([html.Th(etiquetas.get(c, c.replace("_", " "))) for c in df.columns])),
            html.Tbody([
                html.Tr([html.Td(valor(v, c)) for c, v in zip(df.columns, fila)],
                        className="pf-own-row" if "Propio" in fila else "")
                for fila in df.itertuples(index=False, name=None)
            ])
        ]), className="pf-table", style={"overflowX": "auto"})

    def seccion(titulo, *contenido):
        return html.Section([html.H2(titulo), *contenido],
                            className="pf-section")

    def historico(clave):
        return html.Details([
            html.Summary("Apéndice histórico V1 · procedimiento distinto de V2"),
            html.P("V1 conserva sus parámetros y umbrales originales. Boosting es HistGradientBoostingClassifier, no XGBoost. El Brier vacío de SVM indica ausencia de probabilidades. Una precisión alta con pocos positivos predichos no significa alta sensibilidad."),
            anteriores[clave]()
        ])

    @app.server.route("/evidencias-v2/<path:nombre>")
    def evidencia_v2(nombre):
        if nombre not in permitidos:
            return "Archivo no disponible", 404
        return send_from_directory(
            str(base), nombre,
            as_attachment=not nombre.lower().endswith((".png", ".html"))
        )

    intervalos = leer("contrastes/auc_ic95.csv")
    contrastes = leer("contrastes/comparaciones_pareadas.csv")
    referencias = leer("comparacion/resumen_referencias.csv")
    oof = leer("evaluacion/metricas_oof_agrupadas.csv")
    grilla = leer("cobertura/resumen_108_configuraciones.csv")
    propios = leer("comparacion/resumen_modelo_propio_y_ablaciones.csv")
    comparacion_principal = pd.concat([
        propios.loc[propios.familia == "Propio", ["familia", "auc_medio", "sd_auc_folds"]],
        referencias[["familia", "auc_medio", "sd_auc_folds"]].sort_values("auc_medio", ascending=False)
    ], ignore_index=True).merge(intervalos.rename(columns={"modelo": "familia"})[
        ["familia", "ic95_inferior_condicional", "ic95_superior_condicional"]],
        on="familia", how="left", validate="one_to_one")
    casos = leer("shap/casos.csv")
    fidelidad = leer("lime/resumen_fidelidad.csv")

    def grafico_auc():
        d = intervalos.sort_values("auc_medio_folds")
        f = go.Figure(go.Scatter(
            x=d.auc_medio_folds, y=d.modelo.map(lambda x: nombres.get(x, x)), mode="markers",
            marker=dict(size=[14 if x == "Propio" else 9 for x in d.modelo],
                        color=["#b76524" if x == "Propio" else "#147d92" for x in d.modelo]),
            error_x=dict(
                type="data", symmetric=False,
                array=d.ic95_superior_condicional - d.auc_medio_folds,
                arrayminus=d.auc_medio_folds - d.ic95_inferior_condicional
            )
        ))
        f.update_layout(
            template="plotly_white", height=500,
            title="AUC medio por fold e IC del 95 % condicional",
            xaxis_title="AUC medio · eje ampliado para lectura",
            yaxis_title="", margin=dict(l=170, r=30, t=65, b=55)
        )
        return f

    def grafico_diferencias():
        d = contrastes
        f = go.Figure(go.Scatter(
            x=d.delta_auc_medio, y=d.comparacion.map(lambda x: "Modelo propio − " + nombres.get(x.split(" - ")[1], x.split(" - ")[1])), mode="markers",
            marker=dict(size=9, color="#147d92"),
            error_x=dict(
                type="data", symmetric=False,
                array=d.ic95_superior_condicional - d.delta_auc_medio,
                arrayminus=d.delta_auc_medio - d.ic95_inferior_condicional
            )
        ))
        f.add_vline(x=0, line_dash="dash", line_color="#777777")
        f.update_layout(
            template="plotly_white", height=490,
            title="Diferencias de AUC: modelo propio menos comparador",
            xaxis_title="Diferencia de AUC · IC individuales, no simultáneos",
            margin=dict(l=220, r=30, t=65, b=60)
        )
        return f

    def modelo():
        return html.Div([
            anteriores["modelo"](),
            seccion("Interpretación complementaria: XGBoost",
                html.P(
                    "Se explica el XGBoost del fold externo 0 seleccionado por "
                    "validación interna. Estas explicaciones corresponden a una "
                    "referencia, no al modelo propio."
                ),
                imagen("shap/shap_global.png",
                       "SHAP global: media de contribuciones absolutas en 500 "
                       "filas externas. Escala log-odds; columnas transformadas."),
                html.P(
                    "La magnitud absoluta no indica la dirección del efecto. "
                    "smoking_N es una columna codificada y no debe interpretarse "
                    "sin consultar el diccionario. Las asociaciones no son causales."
                ),
                html.Label("Caso externo para comparar SHAP y LIME",
                           htmlFor="v2-caso"),
                dcc.Dropdown(
                    id="v2-caso", clearable=False,
                    options=[
                        {"label": f"{r.tipo.replace('_', ' ')} · ID {int(r.row_id)}",
                         "value": int(r.row_id)}
                        for r in casos.itertuples()
                    ],
                    value=73
                ),
                html.Div(id="v2-explicacion"),
                html.Details([
                    html.Summary("Fidelidad de LIME en tres semillas"),
                    tabla(fidelidad),
                    html.P(
                        "El R² se calcula en la vecindad sintética. No mide "
                        "validez clínica. La semilla 42 se fijó para las figuras; "
                        "no se eligió por obtener el mejor ajuste."
                    )
                ]),
                enlace("shap/metodo.json", "Método SHAP"),
                enlace("lime/metodo.json", "Método LIME")
            )
        ])

    def evaluacion():
        return seccion("4. Evaluación V2 e incertidumbre",
            html.Div([
                html.Strong("540 evaluaciones de folds · 108 configuraciones"),
                html.P(
                    "Siete familias, cuatro optimizadores y cuatro estrategias "
                    "de balanceo. KNN ponderado no es aplicable: cuatro "
                    "combinaciones nominales quedan excluidas."
                )
            ], className="pf-finding"),
            html.P(
                "Se conservan cinco folds externos y tres internos, agrupados "
                "por perfiles. Grid, Random, TPE y búsqueda genética usan un "
                "presupuesto nominal de 16 propuestas por búsqueda; las "
                "configuraciones únicas pueden diferir. Las 108 configuraciones "
                "son combinaciones de familia, optimizador y balanceo, no el "
                "número total de candidatos entrenados."
            ),
            html.P(
                "Las referencias finales eligen optimizador y balanceo por AUC "
                "interno dentro de cada fold. La regla se documentó después de "
                "consultar resultados V2: toda esta evaluación es exploratoria."
            ),
            html.P(
                "SMOTE y ADASYN se aplicaron solo al entrenamiento. En el "
                "espacio codificado pueden producir categorías fraccionarias. "
                "La ponderación utiliza pesos de clase o de observación."
            ),
            html.P(
                "La calibración sigmoide y el umbral Youden se ajustaron dentro "
                "del entrenamiento externo, usando predicciones internas. "
                "Youden es una regla estadística: no establece costos clínicos "
                "ni valida un criterio de derivación odontológica."
            ),
            dcc.Graph(figure=grafico_auc()),
            html.P(
                "Los intervalos usan 2.000 réplicas de perfiles completos dentro "
                "de cada fold, sin estratificación por clase, con pesos compartidos "
                "entre modelos. El estadístico es la media del AUC de los folds. "
                "Son intervalos condicionales a modelos fijos, no simultáneos."
            ),
            html.P(
                "No incorporan toda la variabilidad del entrenamiento ni su "
                "dependencia por solapamiento. Los diez contrastes usan bootstrap "
                "centrado aproximado y corrección Holm. Ese ajuste no elimina "
                "la exploración previa ni demuestra equivalencia."
            ),
            enlace("contrastes/metodo.json", "Método del bootstrap V2"),
            enlace("comparacion/decisiones_solo_validacion_interna.csv",
                   "Selecciones internas"),
            enlace("cobertura/auditoria.json", "Auditoría de cobertura"),
            historico("evaluacion")
        )

    def resultados():
        return seccion("5. Resultados V2, calibración y errores",
            html.Div([
                html.Strong("El modelo propio alcanzó un AUC medio de 0,6051"),
                html.P("Comparé mi propuesta con siete familias de referencia. "
                       "El desempeño fue cercano al de la regresión logística (0,6059). "
                       "Los contrastes no mostraron una mejora frente a las referencias con mayor AUC.")
            ], className="pf-finding"),
            html.H3("Modelo propio y referencias: comparación principal"),
            tabla(comparacion_principal),
            html.P("Presento la media del AUC de los cinco folds externos. La desviación "
                   "estándar (DE) describe su dispersión; los intervalos del 95 % proceden "
                   "del bootstrap condicional. Las referencias se eligieron por validación interna."),
            html.Details([
                html.Summary("Sensibilidad, especificidad y otras métricas de las referencias"),
                tabla(referencias),
                html.P("VPP: valor predictivo positivo. Estas cifras usan umbrales Youden internos. "
                       "El modelo propio conserva el umbral descriptivo de 0,5 de V1; "
                       "no los comparo como si emplearan la misma regla de decisión.")
            ]),
            html.H3("Discriminación, calibración y errores"),
            html.P("Las curvas reúnen las predicciones externas de los cinco folds (OOF). "
                   "Su AUC puede diferir de la media anterior. Incorporé las predicciones "
                   "guardadas del modelo propio a ROC y precisión–recall, sin volver a entrenarlo."),
            html.Label("Figura de evaluación", htmlFor="v2-figura"),
            dcc.Dropdown(id="v2-figura", clearable=False, value="roc", options=[
                {"label": "ROC · modelo propio y referencias", "value": "roc"},
                {"label": "Precisión–recall · modelo propio y referencias", "value": "pr"},
                {"label": "Calibración · referencias V2", "value": "calibracion"},
                {"label": "Matrices de confusión · referencias V2", "value": "matrices"}
            ]),
            html.Div(id="v2-figura-contenido"),
            html.Details([
                html.Summary("Métricas OOF: umbral 0,5 frente a Youden interno"),
                tabla(oof),
                enlace("evaluacion/metricas_oof_agrupadas.csv", "Descargar métricas agrupadas")
            ]),
            html.H3("Modelo propio frente a referencias y ablaciones"),
            dcc.Graph(figure=grafico_diferencias()),
            html.Details([html.Summary("Ver los diez contrastes e intervalos"), tabla(contrastes)]),
            html.P(
                "Encontré diferencias favorables frente a KNN y GaussianNB después del ajuste de Holm en este "
                "análisis exploratorio condicional. No se demostró mejora frente "
                "a logística, SVM, XGBoost ni Random Forest. El intervalo individual "
                "frente al árbol excluye cero, pero su p ajustado no es significativo."
            ),
            html.Details([
                html.Summary("Explorar las 108 configuraciones"),
                html.P(
                    "Comparación descriptiva. El mayor AUC externo observado "
                    "no se utiliza aquí como selección confirmatoria."
                ),
                dcc.Dropdown(
                    id="v2-familia",
                    options=[{"label": nombres.get(x, x), "value": x}
                             for x in sorted(grilla.familia.unique())],
                    value="Logistica_L1_L2", clearable=False
                ),
                html.Div(id="v2-tabla-configuraciones")
            ]),
            enlace("contrastes/comparaciones_pareadas.csv", "Descargar contrastes"),
            enlace("cobertura/resumen_108_configuraciones.csv",
                   "Descargar las 108 configuraciones"),
            historico("resultados")
        )

    def conclusiones():
        return seccion("6. Conclusiones y alcance del estudio",
            html.H3("Respuesta a la pregunta de investigación"),
            html.P(
                "Los datos de chequeo general aportaron discriminación limitada "
                "de caries registrada. El modelo propio obtuvo AUC medio 0,6051; "
                "las referencias seleccionadas internamente alcanzaron 0,6059 "
                "en logística y 0,6055 en XGBoost."
            ),
            html.H3("Qué aportó la implementación"),
            html.P(
                "Desarrollé y verifiqué un modelo con splines, selección de "
                "interacciones y una pérdida mixta de clasificación y ranking. "
                "Evalué el aporte de sus componentes mediante ablaciones. La implementación es propia, "
                "pero la novedad científica de la combinación no está establecida."
            ),
            html.H3("Qué permite concluir la comparación"),
            html.P(
                "El análisis condicional exploratorio favorece al modelo propio "
                "frente a KNN y GaussianNB después de Holm. No demuestra mejora "
                "frente a las referencias con mayor AUC. Las ablaciones no "
                "demostraron una ventaja de incorporar interacciones o ranking. "
                "Estos resultados no me permiten afirmar equivalencia ni superioridad confirmatoria."
            ),
            html.H3("Umbrales y utilidad potencial"),
            html.P(
                "En V2, los umbrales internos permiten sensibilidades aproximadas "
                "de 47–55 %, con especificidades de 58–69 % entre referencias. "
                "Esto modifica el compromiso entre errores, no demuestra una "
                "mejora del AUC ni acredita utilidad clínica. En V1, el modelo "
                "propio conservó su evaluación descriptiva con umbral 0,5."
            ),
            html.H3("Interpretación y límites"),
            html.P(
                "SHAP reconstruye el margen del XGBoost analizado. LIME aproxima "
                "su probabilidad calibrada; presenta menor fidelidad en los casos "
                "negativos. En el verdadero negativo ID 2, el error absoluto "
                "medio fue de 6,03 puntos porcentuales."
            ),
            html.P(
                "Se clasifica caries registrada, no caries futura. Agrupar "
                "perfiles no acredita independencia por paciente. El test histórico "
                "ya fue consultado. Los intervalos no incorporan toda la "
                "variabilidad de reentrenamiento. La procedencia clínica, la "
                "definición diagnóstica y la disponibilidad temporal requieren "
                "verificación adicional."
            ),
            html.H3("Trabajo futuro"),
            html.P(
                "Como continuación de este trabajo, propongo verificar los metadatos y ampliar la revisión de antecedentes, evaluar la selección de "
                "interacciones con residuos fuera de muestra y realizar una "
                "validación independiente. Una nueva variante se documentaría "
                "como exploratoria; no reemplazaría retrospectivamente V1."
            ),
            enlace("LEEME.md", "Resumen metodológico"),
            enlace("manifest.json", "Trazabilidad de evidencias")
        )

    @app.callback(Output("v2-figura-contenido", "children"), Input("v2-figura", "value"))
    def figura_evaluacion(clave):
        if clave in ("roc", "pr"):
            texto = ("ROC del modelo propio y las referencias; predicciones OOF conservadas." if clave == "roc"
                     else "Precisión–recall del modelo propio y las referencias. La línea horizontal indica la prevalencia.")
            return html.Figure([
                html.Img(src=f"/evidencias/{clave}_propio_referencias.png", alt=texto),
                html.Figcaption(texto + " La comparación inferencial utiliza el AUC medio por fold, no esta curva agrupada.")
            ], className="pf-eval-figure")
        if clave == "calibracion":
            return html.Div([
                imagen("evaluacion/calibracion_referencias.png", "Calibración de las referencias V2 en diez intervalos fijos."),
                html.P("Un ECE bajo no garantiza buena discriminación. Interpreto esta medida junto con Brier, AUC y precisión promedio.")
            ], className="pf-eval-figure")
        if clave == "matrices":
            return html.Div([
                imagen("evaluacion/matrices_umbrales_internos.png", "Matrices de las referencias V2: porcentajes por clase observada."),
                html.P("Cada fold conserva el umbral elegido internamente. Estas matrices no se utilizaron para ajustarlo.")
            ], className="pf-eval-figure")
        return html.P("Selecciona una figura disponible.")

    @app.callback(Output("v2-tabla-configuraciones", "children"),
                  Input("v2-familia", "value"))
    def configuraciones(familia):
        return tabla(grilla.loc[grilla.familia == familia].sort_values(
            "auc_medio", ascending=False))

    @app.callback(Output("v2-explicacion", "children"),
                  Input("v2-caso", "value"))
    def explicar(rid):
        sub = casos.loc[casos.row_id == rid]
        if len(sub) != 1:
            return html.P("Selecciona un caso disponible.")
        r = sub.iloc[0]
        tipo, rid = r["tipo"], int(r["row_id"])
        return html.Div([
            html.P(
                f"ID {rid} · clase observada {int(r['actual'])} · "
                f"predicción {int(r['prediction'])}. Probabilidad original "
                f"{r['probability_original']:.4f}; calibrada "
                f"{r['probability_sigmoide']:.4f}; umbral interno "
                f"{r['umbral_interno']:.4f}."
            ),
            imagen(f"shap/shap_{tipo}_{rid}.png",
                   "SHAP: contribuciones al margen original, escala log-odds."),
            imagen(f"lime/lime_{tipo}_{rid}.png",
                   "LIME: aproximación local de probabilidad calibrada, semilla 42."),
            tabla(fidelidad.loc[fidelidad.row_id == rid]),
            html.P(
                "Estas explicaciones usan escalas diferentes. Sus pesos no "
                "son directamente comparables ni representan efectos causales."
            ),
            enlace(f"lime/lime_{tipo}_{rid}.html", "Abrir explicación LIME")
        ])

    secciones.update(modelo=modelo, evaluacion=evaluacion,
                     resultados=resultados, conclusiones=conclusiones)

    # Actualizar solo la cifra de cobertura de la portada.
    def actualizar_tarjeta(nodo):
        if getattr(nodo, "className", None) == "pf-metrics":
            hijos = list(nodo.children)
            if len(hijos) == 4:
                hijos[-1] = html.Div([
                    html.Strong("108"),
                    html.Span("Configuraciones V2 · 540 evaluaciones de folds")
                ])
                nodo.children = hijos
        hijos = getattr(nodo, "children", None)
        if isinstance(hijos, (list, tuple)):
            for hijo in hijos:
                actualizar_tarjeta(hijo)
        elif hasattr(hijos, "children"):
            actualizar_tarjeta(hijos)

    actualizar_tarjeta(app.layout)
    return {"configuraciones": configuraciones, "explicar": explicar, "figura": figura_evaluacion}
