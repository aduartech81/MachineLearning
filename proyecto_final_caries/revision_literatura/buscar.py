"""Búsqueda preliminar reproducible en índices públicos; no demuestra novedad."""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen,Request
from datetime import datetime,timezone
import json,time,hashlib,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
QUERIES=[('dental caries','generalized additive model'),('dental caries','pairwise ranking loss'),('dental caries','spline logistic model'),('caries dental','modelo aditivo ranking'),('generalized additive model','pairwise logistic AUC optimization')]
def main():
 out=BASE/'resultados';out.mkdir(exist_ok=False);logs=[]
 for database in ['PubMed','Crossref','arXiv']:
  for i,(application,method) in enumerate(QUERIES,1):
   text=application+' '+method;date=datetime.now(timezone.utc).isoformat();record=dict(base=database,id_consulta=i,cadena=text,fecha_utc=date,alcance='Metadatos de hasta 10 resultados; sin revisión sistemática completa de texto integral')
   if database=='PubMed':url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urlencode(dict(db='pubmed',term=text,retmode='json',retmax=10))
   elif database=='Crossref':url='https://api.crossref.org/works?'+urlencode({'query':text,'rows':10})
   else:url='https://export.arxiv.org/api/query?'+urlencode(dict(search_query='all:"'+application+'" AND all:"'+method+'"',start=0,max_results=10))
   record['url']=url
   try:
    raw=urlopen(Request(url,headers={'User-Agent':'AcademicMLProject/1.0 (literature metadata review)'}),timeout=25).read();ext='xml' if database=='arXiv' else 'json';file=out/f'{database}_{i}.{ext}';file.write_bytes(raw);record['sha256']=hashlib.sha256(raw).hexdigest();record['estado']='Consulta respondida'
    if database=='Crossref':
     data=json.loads(raw)['message'];record['total_indice_no_pertinencia']=data['total-results'];record['resultados']=[{'titulo':x.get('title',[''])[0],'DOI':x.get('DOI'),'url':x.get('URL')} for x in data['items']]
    elif database=='PubMed':
     data=json.loads(raw)['esearchresult'];record['total_indice_no_pertinencia']=int(data['count']);record['identificadores']=data['idlist']
    else:
     data=ET.fromstring(raw);record['resultados']=[dict(titulo=e.findtext('{http://www.w3.org/2005/Atom}title'),url=e.findtext('{http://www.w3.org/2005/Atom}id')) for e in data.findall('{http://www.w3.org/2005/Atom}entry')]
   except Exception as e:record['estado']='Acceso fallido; no contabilizar como búsqueda completada';record['error']=str(e)
   logs.append(record);(out/'registro_consultas.json').write_text(json.dumps(logs,indent=2,ensure_ascii=False));print(database,i,record['estado'],flush=True);time.sleep(.4)
 print('REGISTRO FINALIZADO',flush=True)
if __name__=='__main__':main()
