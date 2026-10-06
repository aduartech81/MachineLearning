"""Dashboard del segundo entregable: exactamente tres pestañas."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from ml_common import evaluate

ROOT = Path(__file__).parent
summary = json.loads((ROOT/'results/summary.json').read_text())
train = pd.read_csv(ROOT/'results/train.csv')
pred = pd.read_csv(ROOT/'results/predictions.csv')
px.defaults.template = 'plotly_white'
COLOR = ['#147d92', '#df7955']
app = Dash(__name__, assets_folder=str(ROOT/'assets'))
server = app.server
app.title = 'Caries dental | Andrés Duarte'


def graph(fig):
    fig.update_layout(margin=dict(l=40, r=25, t=55, b=40), font_family='Arial')
    return dcc.Graph(figure=fig, config={'displaylogo': False})


def note(text):
    return html.P(text, className='note')


def card(value, label):
    return html.Div([html.Strong(value), html.Span(label)], className='card')


def context():
    counts = pd.DataFrame({'Estado': ['Sin caries registrada', 'Con caries registrada'],
                           'Registros': [summary['rows']-summary['positive_rows'], summary['positive_rows']]})
    return html.Div([
        html.H2('Clasificar caries registrada a partir de datos de chequeo general'),
        html.P('Proyecto académico de clasificación binaria. La variable dental caries representa '
               'el estado registrado durante un chequeo; no predice aparición futura de caries.'),
        html.Div([card(f"{summary['rows']:,}", 'registros'),
                  card(f"{summary['prevalence']:.1%}", 'con caries'),
                  card(str(len(summary['features'])), 'predictores'),
                  card('Logística', 'modelo base')], className='cards'),
        graph(px.bar(counts, x='Estado', y='Registros', color='Estado', color_discrete_sequence=COLOR,
                     title='Distribución de la variable objetivo')),
        note('La clase negativa es mayoritaria. La exactitud debe acompañarse de sensibilidad, '
             'ROC AUC y Average Precision; predecir siempre ausencia de caries no identifica casos positivos.'),
        html.H3('Fuente y alcance'),
        html.P(['Body Signal of Smoking, copia pública del archivo smoking.csv. ',
                html.A('Fuente utilizada', href='https://github.com/aliabdallah7/smoking-detection', target='_blank'),
                '. Los conteos coinciden con el primer entregable; la identidad con el archivo original '
                'del estudiante no está confirmada por checksum.']),
        html.P('Se excluyen ID, oral (constante) y tartar. Esta última variable requiere exploración '
               'oral y no se presume disponible antes del examen que registra caries.'),
        html.H3('Diseño de evaluación'),
        html.P(f"Entrenamiento: {summary['train_rows']:,}; prueba: {summary['test_rows']:,}. "
               'Partición estratificada por perfiles únicos, conservando sus copias en el mismo conjunto. '
               'EDA exclusivamente sobre entrenamiento. El conjunto de prueba ya usado anteriormente '
               'no puede considerarse una validación externa nueva.'),
        note('La base contiene 11.140 filas repetidas sin ID. La partición protege frente a copias '
             'idénticas, pero no garantiza independencia por persona: no hay identificador longitudinal verificado.'),
        html.P('Autor: Andrés Duarte Chaves · Docente: Dr. Lihki Rubio · Maestría en Ingeniería Biomédica')])


def eda():
    nums = train.select_dtypes(include=np.number).columns.difference(['ID', 'dental caries']).tolist()
    return html.Div([
        html.H2('Exploración del conjunto de entrenamiento'),
        html.Div([html.Div([html.Label('Variable numérica'),
                            dcc.Dropdown(nums, 'age', id='variable')]),
                  html.Div([html.Label('Sexo registrado'),
                            dcc.Dropdown(['Todos', 'F', 'M'], 'Todos', id='gender')])], className='filters'),
        html.Div(id='eda-cards', className='cards'),
        html.Div([dcc.Graph(id='histogram'), dcc.Graph(id='boxplot')], className='grid'),
        note('El histograma muestra frecuencias y el boxplot resume mediana, dispersión y valores extremos. '
             'Los filtros cambian la población descrita; diferencias entre grupos no implican causalidad.'),
        dcc.Graph(id='association'),
        note('Las proporciones por edad se calculan en entrenamiento y dependen del filtro de sexo. '
             'Son asociaciones transversales; no estiman incidencia ni efectos causales.'),
        graph(px.imshow(train[summary['features']].select_dtypes(include=np.number).corr(),
                        zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                        title='Correlaciones numéricas de entrenamiento')),
        note('Correlaciones altas señalan redundancia entre predictores. No prueban relación causal '
             'ni ausencia de fuga de información. Esta matriz usa todo el entrenamiento y no cambia con los filtros.'),
        html.P(f"Faltantes en el archivo recibido: {summary['missing_values']}. No se asume un mecanismo MAR. "
               'Las filas extremas se conservan; una distribución por sí sola no permite diagnosticar enfermedades.')])


def models():
    fpr, tpr, _ = roc_curve(pred.actual, pred.probability)
    roc = go.Figure([go.Scatter(x=fpr, y=tpr, name='Regresión logística', line_color=COLOR[0]),
                     go.Scatter(x=[0, 1], y=[0, 1], name='Azar', line_dash='dash')])
    roc.update_layout(title='Curva ROC — prueba interna', xaxis_title='Tasa de falsos positivos',
                      yaxis_title='Sensibilidad')
    precision, recall, _ = precision_recall_curve(pred.actual, pred.probability)
    pr = go.Figure(go.Scatter(x=recall, y=precision, name='Logística', line_color=COLOR[0]))
    pr.add_hline(y=summary['test_prevalence'], line_dash='dash')
    pr.update_layout(title='Precision–Recall — prueba interna', xaxis_title='Recall', yaxis_title='Precision')
    fraction, mean = calibration_curve(pred.actual, pred.probability, n_bins=10, strategy='quantile')
    cal = go.Figure([go.Scatter(x=mean, y=fraction, mode='lines+markers', name='Modelo'),
                    go.Scatter(x=[0, 1], y=[0, 1], name='Referencia', line_dash='dash')])
    cal.update_layout(title='Calibración por deciles de probabilidad',
                      xaxis_title='Probabilidad media', yaxis_title='Fracción positiva observada')
    comparison = pd.DataFrame([dict(Modelo='Logística', **summary['logistic']),
                               dict(Modelo='Dummy (prior)', **summary['dummy'])])
    columns = ['Modelo', 'roc_auc', 'average_precision', 'accuracy', 'recall', 'f1', 'brier']
    return html.Div([
        html.H2('Modelo base: regresión logística frente a Dummy'),
        html.P('Imputación, estandarización y codificación categórica dentro de un Pipeline. '
               'Validación cruzada en perfiles únicos de entrenamiento; modelo final ajustado con todas '
               'las filas de entrenamiento. La duplicación puede ponderar algunos perfiles más que otros.'),
        html.Div([card(f"{summary['logistic']['roc_auc']:.3f}", 'ROC AUC'),
                  card(f"{summary['logistic']['average_precision']:.3f}", 'Average Precision'),
                  card(f"{summary['cv_auc_mean']:.3f} ± {summary['cv_auc_sd']:.3f}", 'AUC CV: media ± DE')], className='cards'),
        html.Table([html.Thead(html.Tr([html.Th(c) for c in columns])),
                    html.Tbody([html.Tr([html.Td(row[c] if c == 'Modelo' else f'{row[c]:.4f}')
                                         for c in columns]) for _, row in comparison.iterrows()])]),
        note('La tabla usa umbral 0,5. ROC AUC describe discriminación; AP resume precision–recall '
             'y no equivale exactamente al área trapezoidal de esa curva. Brier evalúa el error de probabilidades.'),
        html.Div([graph(roc), graph(pr)], className='grid'),
        note('ROC y PR muestran el comportamiento al variar el corte. El nivel de prevalencia es la '
             'referencia de PR. Una AUC superior a 0,5 no implica utilidad clínica suficiente.'),
        html.Label('Explorar umbral (no se usa para optimizar el modelo sobre test)'),
        dcc.Slider(0.05, 0.95, 0.05, value=0.5, marks={0.1: '0,1', 0.5: '0,5', 0.9: '0,9'}, id='threshold'),
        html.Div(id='threshold-cards', className='cards'), dcc.Graph(id='confusion'),
        note('El control es una demostración retrospectiva. Elegir un umbral operativo requiere validación '
             'sobre entrenamiento y un criterio explícito de costos; no debe elegirse por el mejor resultado en test.'),
        graph(cal), note('Cercanía a la diagonal sugiere acuerdo entre probabilidades y frecuencias. '
                         'La curva no confirma calibración clínica ni transportabilidad a Colombia.')])


app.layout = html.Div([
    html.Header([html.P('MACHINE LEARNING · ENTREGABLE 2', className='eyebrow'),
                 html.H1('Caries dental'), html.P('Datos, exploración y evaluación de un modelo base')]),
    dcc.Tabs(id='tabs', value='context', children=[
        dcc.Tab(label='1 · Contexto del problema', value='context'),
        dcc.Tab(label='2 · EDA', value='eda'),
        dcc.Tab(label='3 · Modelos base', value='models')]),
    html.Main(id='content')], className='shell')
app.validation_layout = html.Div([app.layout, context(), eda(), models()])


@app.callback(Output('content', 'children'), Input('tabs', 'value'))
def tab_content(value):
    return {'context': context, 'eda': eda, 'models': models}[value]()


@app.callback(Output('histogram', 'figure'), Output('boxplot', 'figure'),
              Output('association', 'figure'), Output('eda-cards', 'children'),
              Input('variable', 'value'), Input('gender', 'value'))
def update_eda(variable, gender):
    subset = train if gender == 'Todos' else train.loc[train.gender == gender]
    subset = subset.copy()
    subset['Estado'] = subset['dental caries'].map({0: 'Sin caries', 1: 'Con caries'})
    hist = px.histogram(subset, x=variable, color='Estado', barmode='overlay',
                        opacity=0.65, nbins=35, color_discrete_sequence=COLOR,
                        title=f'Distribución de {variable}')
    box = px.box(subset, x='Estado', y=variable, color='Estado', points=False,
                 color_discrete_sequence=COLOR, title=f'Dispersión de {variable}')
    rates = subset.groupby('age', as_index=False)['dental caries'].agg(['mean', 'count']).reset_index()
    rates['Porcentaje'] = rates['mean']*100
    assoc = px.bar(rates, x='age', y='Porcentaje', hover_data=['count'],
                   title='Proporción de caries por edad registrada')
    return hist, box, assoc, [card(f'{len(subset):,}', 'filas filtradas'),
                              card(f"{subset['dental caries'].mean():.1%}", 'con caries en el filtro')]


@app.callback(Output('confusion', 'figure'), Output('threshold-cards', 'children'),
              Input('threshold', 'value'))
def update_threshold(threshold):
    m = evaluate(pred.actual, pred.probability, threshold)
    fig = px.imshow([[m['tn'], m['fp']], [m['fn'], m['tp']]], text_auto=True,
                    x=['Predicción 0', 'Predicción 1'], y=['Real 0', 'Real 1'],
                    color_continuous_scale='Blues', title=f'Matriz de confusión · corte {threshold:.2f}')
    return fig, [card(f"{m['recall']:.1%}", 'sensibilidad'),
                 card(f"{m['specificity']:.1%}", 'especificidad'),
                 card(str(m['fn']), 'falsos negativos'), card(f"{m['f1']:.3f}", 'F1')]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
