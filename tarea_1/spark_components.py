from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable,DefaultParamsWritable
from pyspark.storagelevel import StorageLevel

class CacheVectors(Transformer,DefaultParamsReadable,DefaultParamsWritable):
    """Materialización de vectores dentro de cada fold antes de ajustar."""
    def _transform(self,dataset):
        cached=dataset.persist(StorageLevel.MEMORY_AND_DISK)
        cached.count()
        return cached
