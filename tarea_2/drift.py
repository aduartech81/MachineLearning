"""Comparación ilustrativa train-test y control positivo de deriva simulado."""
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).parent/'results'


def main():
    reference = pd.read_csv(ROOT/'reference.csv')
    current = pd.read_csv(ROOT/'current.csv')
    for name, data in [('drift_report', current),
                       ('drift_simulated', current.assign(Age=current.Age+20))]:
        result = Report([DataDriftPreset()]).run(current_data=data, reference_data=reference)
        result.save_html(str(ROOT/f'{name}.html'))
        result.save_json(str(ROOT/f'{name}.json'))
        print('Creado', name)


if __name__ == '__main__':
    main()
