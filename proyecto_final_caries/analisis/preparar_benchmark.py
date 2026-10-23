"""Recuperar solo las 35 selecciones publicadas desde el ZIP ya existente."""
import argparse
import csv
import hashlib
from pathlib import Path
import zipfile


def main():
    """Verificar huellas antes de crear archivos; nunca sobrescribir diferencias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip_evidencias', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    decisiones = root / 'proyecto_final_caries/presentacion/evidencias/revision_v2_20261020_154421_004423/comparacion/decisiones_solo_validacion_interna.csv'
    with decisiones.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 35 and len({(r['familia'], r['fold']) for r in rows}) == 35
    pendientes = []
    with zipfile.ZipFile(args.zip_evidencias) as z:
        for row in rows:
            name = row['seleccion_origen']
            dest = (root / name).resolve()
            if not dest.is_relative_to(root):
                raise ValueError('Ruta fuera de la copia de trabajo.')
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != row['seleccion_sha256']:
                raise ValueError(f'Huella diferente: {name}')
            if dest.exists():
                if dest.read_bytes() != data:
                    raise ValueError(f'Archivo existente diferente: {name}')
            else:
                pendientes.append((dest, data))
    for dest, data in pendientes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(data)
    print(f'35 selecciones verificadas; {len(pendientes)} recuperadas. Sin entrenamiento.')


if __name__ == '__main__':
    main()
