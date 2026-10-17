from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable,DefaultParamsWritable
from pyspark.storagelevel import StorageLevel

_cached_vectors = None

class CacheVectors(Transformer,DefaultParamsReadable,DefaultParamsWritable):
    """Materializa sólo columnas necesarias y libera el cache anterior (CV serial)."""
    def _transform(self,dataset):
        global _cached_vectors
        if _cached_vectors is not None:
            _cached_vectors.unpersist(blocking=True)
        columns=[c for c in ['features','raw_features','label','row_id','fold','lime_id'] if c in dataset.columns]
        cached=dataset.select(*columns).persist(StorageLevel.MEMORY_AND_DISK)
        cached.count()
        _cached_vectors=cached
        return cached
