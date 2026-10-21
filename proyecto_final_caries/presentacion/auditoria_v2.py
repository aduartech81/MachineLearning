"""Ampliación documental a partir de evidencias recuperadas; sin entrenamientos."""
from dash import html, dcc
import pandas as pd

def integrar(app, secciones, data):
    base = data / "recuperacion_v2"
    previo = secciones["evaluacion"]
    def evaluacion():
        contenido = previo()
        tabla = pd.read_csv(base / "presupuesto_efectivo.csv")
        tabla["segundos"] = tabla["segundos"].round(2)
        contenido.children.append(html.Section([
            dcc.Markdown((base / "interpretacion.md").read_text()),
            html.H3("Presupuesto efectivo registrado"),
            html.Table([html.Thead(html.Tr([html.Th(c) for c in tabla.columns])),
                        html.Tbody([html.Tr([html.Td(str(v)) for v in fila]) for fila in tabla.itertuples(index=False,name=None)])]),
            html.Img(src="/auditoria-v2/calibracion_antes_despues.png", alt="Brier y log-loss antes y después de calibrar", style={"width":"100%"}),
            html.Details([html.Summary("Descargar código, historiales y comprobaciones"),
                         html.Ul([html.Li(html.A(p.name,href="/auditoria-v2/"+p.name)) for p in sorted(base.iterdir()) if p.is_file()])])
        ], className="pf-section"))
        return contenido
    secciones["evaluacion"] = evaluacion
