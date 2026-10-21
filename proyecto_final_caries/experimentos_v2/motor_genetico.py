
"""Motor genético secuencial: población 4, descendencia 4, presupuesto 16."""
import copy
import math
import random
from deap import base, tools

class Aptitud(base.Fitness):
    weights = (1.0,)

class Individuo(list):
    def __init__(self, genes):
        super().__init__(genes)
        self.fitness = Aptitud()

class Propuesta:
    def __init__(self, individuo, dominios, numero):
        self.individuo = individuo
        self.numero = numero
        self.generacion = numero // 4
        self.genes = list(individuo)
        self.parametros = {}
        for (nombre, dominio), gen in zip(dominios.items(), individuo):
            if dominio["tipo"] == "categorico":
                valor = dominio["valores"][int(gen)]
            else:
                valor = min(dominio["max"], max(dominio["min"], math.exp(gen)))
            self.parametros[nombre] = valor

    def suggest_categorical(self, nombre, valores):
        valor = self.parametros[nombre]
        assert valor in valores
        return valor

    def suggest_float(self, nombre, minimo, maximo, log=False):
        valor = self.parametros[nombre]
        assert minimo <= valor <= maximo
        return valor

class EstudioGenetico:
    def __init__(self, dominios, seed):
        self.dominios = dominios
        self.estado_rng = random.Random(seed).getstate()
        self.numero = 0
        self.padres = []
        self.lote = []
        self.pendiente = None

    def _gen(self, dominio):
        if dominio["tipo"] == "categorico":
            return random.randrange(len(dominio["valores"]))
        if dominio["tipo"] == "real_log":
            return random.uniform(math.log(dominio["min"]), math.log(dominio["max"]))
        raise ValueError(dominio)

    def _nuevo_lote(self):
        if not self.lote:
            self.lote = [
                Individuo([self._gen(d) for d in self.dominios.values()])
                for _ in range(4)
            ]
            return
        assert all(i.fitness.valid for i in self.lote)
        self.padres = tools.selBest(self.padres + self.lote, 4)
        hijos = []
        while len(hijos) < 4:
            a, b = [
                copy.deepcopy(i)
                for i in tools.selTournament(self.padres, 2, tournsize=2)
            ]
            if random.random() < 0.8:
                tools.cxUniform(a, b, indpb=0.5)
            for hijo in (a, b):
                if random.random() < 0.5:
                    for j, dominio in enumerate(self.dominios.values()):
                        if random.random() < 1.0 / len(hijo):
                            if dominio["tipo"] == "categorico":
                                hijo[j] = self._gen(dominio)
                            else:
                                inferior = math.log(dominio["min"])
                                superior = math.log(dominio["max"])
                                hijo[j] = min(superior, max(
                                    inferior,
                                    hijo[j] + random.gauss(
                                        0, 0.2 * (superior - inferior)
                                    )
                                ))
                del hijo.fitness.values
                hijos.append(hijo)
        self.lote = hijos

    def ask(self):
        if self.pendiente is not None:
            raise RuntimeError("Primero registrar el resultado anterior.")
        if self.numero >= 16:
            raise RuntimeError("Presupuesto agotado.")
        # Aislar la aleatoriedad del motor de la usada por los clasificadores.
        estado_global = random.getstate()
        random.setstate(self.estado_rng)
        try:
            if self.numero % 4 == 0:
                self._nuevo_lote()
            propuesta = Propuesta(
                self.lote[self.numero % 4], self.dominios, self.numero
            )
        finally:
            self.estado_rng = random.getstate()
            random.setstate(estado_global)
        self.pendiente = propuesta
        return propuesta

    def tell(self, propuesta, valor):
        assert propuesta is self.pendiente
        assert math.isfinite(valor)
        propuesta.individuo.fitness.values = (float(valor),)
        self.numero += 1
        self.pendiente = None
