"""Monta ``amostras_5classes.gpkg``: pontos rotulados em cinco classes de uso do solo.

Classes e origem:

- ``floresta``, ``pastagem`` e ``agricultura`` vêm de ``samples_matogrosso_mod13q1``, o
  conjunto de referência do pacote ``sits`` (INPE/e-sensing), já recortado em
  ``amostras_uso_do_solo_mt.gpkg``: ``Forest`` -> floresta, ``Pasture`` -> pastagem,
  ``Soy_*`` -> agricultura; ``Cerrado`` é descartado.
- ``agua`` e ``urbano`` vêm da Coleção 9 do MapBiomas, ano 2023 (classes 33, "Rio, Lago e
  Oceano", e 24, "Área Urbanizada"), sorteadas dentro da mesma região e confirmadas em
  resolução plena: só entra o ponto cujo entorno de 5x5 pixels (150 m) é todo da classe.

Cada classe fica com no máximo ``ALVO`` locais, sorteados com ``random_state`` fixo; a
coluna ``fonte`` registra a origem de cada ponto.

O raster do MapBiomas é lido remotamente (COG em ``storage.googleapis.com``), sem baixar
o arquivo de 1 GB. Se o GDAL reclamar do certificado, aponte ``CURL_CA_BUNDLE`` para um
``cacert.pem`` num caminho sem acentos (``python -c "import certifi; print(certifi.where())"``).

Uso, na pasta ``examples/casos``::

    python preparar_amostras_5classes.py
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds

AQUI = pathlib.Path(__file__).resolve().parent
ORIGEM_SITS = AQUI / 'amostras_uso_do_solo_mt.gpkg'
DESTINO = AQUI / 'amostras_5classes.gpkg'

MAPBIOMAS = ('/vsicurl/https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/'
             'collection_9/lclu/coverage/brasil_coverage_2023.tif')
FONTE_SITS = 'sits: samples_matogrosso_mod13q1 (INPE/e-sensing)'
FONTE_MAPBIOMAS = 'MapBiomas Colecao 9, cobertura 2023, classe {codigo}'
CODIGOS = {'agua': 33, 'urbano': 24}

# Mesma regiao das amostras do sits: 2 graus em torno do ponto do caso 1.
OESTE, SUL, LESTE, NORTE = -58.0, -15.55, -54.0, -11.55
ALVO = 50
SEMENTE = 42
ESPACAMENTO_MINIMO = 0.01     # ~1,1 km entre pontos sorteados da mesma classe
JANELA = 5                    # entorno conferido em resolucao plena (5 x 30 m)


def amostras_do_sits():
    """Reclassifica o recorte do sits em tres classes e sorteia ate ALVO locais de cada."""
    base = gpd.read_file(ORIGEM_SITS)
    locais = base.drop_duplicates(subset=['longitude', 'latitude'])
    classe = {'Forest': 'floresta', 'Pasture': 'pastagem'}
    locais = locais.assign(
        classe=locais['label'].map(lambda r: classe.get(r, 'agricultura' if r.startswith('Soy_') else None)))
    locais = locais.dropna(subset=['classe'])          # descarta Cerrado

    partes = []
    for nome, grupo in locais.groupby('classe'):
        n = min(ALVO, len(grupo))
        partes.append(grupo.sample(n=n, random_state=SEMENTE))
    saida = pd.concat(partes)
    return pd.DataFrame({'longitude': saida['longitude'].values,
                         'latitude': saida['latitude'].values,
                         'classe': saida['classe'].values,
                         'fonte': FONTE_SITS})


def amostras_do_mapbiomas(classe, codigo, rng):
    """Sorteia pontos da classe no overview e confirma cada um em resolucao plena."""
    with rasterio.open(MAPBIOMAS) as src:
        janela = from_bounds(OESTE, SUL, LESTE, NORTE, src.transform)
        fator = 16
        forma = (int(janela.height // fator), int(janela.width // fator))
        grosso = src.read(1, window=janela, out_shape=forma)

        linhas, colunas = np.where(grosso == codigo)
        ordem = rng.permutation(len(linhas))
        passo_lon = (LESTE - OESTE) / grosso.shape[1]
        passo_lat = (NORTE - SUL) / grosso.shape[0]

        aceitos = []
        for k in ordem:
            lon = OESTE + (colunas[k] + 0.5) * passo_lon
            lat = NORTE - (linhas[k] + 0.5) * passo_lat
            if any(np.hypot(lon - a, lat - b) < ESPACAMENTO_MINIMO for a, b in aceitos):
                continue
            r, c = src.index(lon, lat)
            meio = JANELA // 2
            bloco = src.read(1, window=Window(c - meio, r - meio, JANELA, JANELA))
            if bloco.shape == (JANELA, JANELA) and (bloco == codigo).all():
                aceitos.append((lon, lat))
            if len(aceitos) == ALVO:
                break

    return pd.DataFrame({'longitude': [a for a, _ in aceitos],
                         'latitude': [b for _, b in aceitos],
                         'classe': classe,
                         'fonte': FONTE_MAPBIOMAS.format(codigo=codigo)})


def main():
    rng = np.random.default_rng(SEMENTE)
    partes = [amostras_do_sits()]
    for classe, codigo in CODIGOS.items():
        parte = amostras_do_mapbiomas(classe, codigo, rng)
        print(f'{classe}: {len(parte)} pontos confirmados no MapBiomas')
        partes.append(parte)

    todas = pd.concat(partes, ignore_index=True)
    gdf = gpd.GeoDataFrame(todas, geometry=gpd.points_from_xy(todas['longitude'], todas['latitude']),
                           crs='EPSG:4326')
    gdf.to_file(DESTINO, driver='GPKG')
    print(f'\n{DESTINO.name}: {len(gdf)} pontos')
    print(gdf.groupby(['classe', 'fonte']).size().to_string())


if __name__ == '__main__':
    main()
