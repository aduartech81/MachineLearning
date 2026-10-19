"""Orquestación reanudable; cada fase debe finalizar antes de continuar."""
import os,sys,json,subprocess,datetime
from tarea_1.common import OUT

def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    stages=['prepare','run_sklearn','run_spark','eda','statistics','explain','package_predictions','report']
    if len(sys.argv)>1:stages=stages[stages.index(sys.argv[1]):]
    for stage in stages:
        state=dict(stage=stage,status='running',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (OUT/'execution_state.json').write_text(json.dumps(state,indent=2))
        code=subprocess.call([sys.executable,'-u','-m',f'tarea_1.{stage}'],env=env)
        state['status']='completed' if code==0 else 'failed';state['exit_code']=code
        (OUT/'execution_state.json').write_text(json.dumps(state,indent=2))
        if code:raise SystemExit(code)
    (OUT/'execution_state.json').write_text(json.dumps(dict(status='completed',stages=stages),indent=2))
if __name__=='__main__':main()
