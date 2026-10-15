"""Monta ``amostras_terraclass_5classes.gpkg``: os mesmos locais de
``amostras_5classes.gpkg``, rerrotulados pelo TerraClass via WLTS.

Por que rerrotular em vez de sortear pontos novos: os locais de
``amostras_5classes.gpkg`` já foram escolhidos espalhados pela região e, no caso de
``agua`` e ``urbano``, conferidos num entorno homogêneo de 5x5 pixels. O que se troca
aqui é a *fonte do rótulo*, não a amostragem: sai o par ``sits`` (2000/01 a 2015/16) +
MapBiomas (2023) e entra o TerraClass, mapeamento oficial de 30 m derivado de imagem,
mais próximo em resolução do Sentinel-2 de 10 m deste caso.

A região fica na fronteira dos biomas, então nenhuma das duas coleções do TerraClass
cobre sozinha todos os pontos: ``terraclass_amazonia`` devolve ``No-Data`` fora da
Amazônia Legal e ``terraclass_cerrado`` devolve ``No-info`` fora do Cerrado. As duas são
consultadas e combinadas.

Ano base: 2020, o mais recente que as duas coleções servem em comum pelo WLTS
(``terraclass_amazonia`` já publica 2022; ``terraclass_cerrado`` para em 2020).

As classes originais do TerraClass são reagrupadas nas cinco deste caso por ``MAPA``.
Ponto cuja classe não cai em nenhuma delas (``Outros Usos``, ``Outras Áreas
Edificadas``, ``Mineração``, ``Silvicultura``) ou que as duas coleções rotulam de forma
divergente é descartado, e o descarte é impresso no fim.

Uso, na pasta ``examples/caso1``::

    python preparar_amostras_terraclass.py
"""
import pathlib

import geopandas as gpd
import pandas as pd
import wlts

AQUI = pathlib.Path(__file__).resolve().parent
ORIGEM = AQUI / 'amostras_5classes.gpkg'
DESTINO = AQUI / 'amostras_terraclass_5classes.gpkg'

SERVICO_WLTS = 'https://data.inpe.br/bdc/wlts/v1/'
COLECOES = ('terraclass_amazonia', 'terraclass_cerrado')
ANO_BASE = '2020'

# Classes do TerraClass (Amazonia e Cerrado) reagrupadas nas cinco deste caso.
MAPA = {
    'Vegetação Natural Florestal Primária': 'floresta',
    'Vegetação Natural Florestal Secundária': 'floresta',
    'Vegetação Natural Primária': 'floresta',
    'Vegetação Natural Secundária': 'floresta',

    'Pastagem Herbacea': 'pastagem',
    'Pastagem Arbustica/Arborea': 'pastagem',
    'Pastagem': 'pastagem',

    'Cultura Agricola Temporaria de 1 Ciclo': 'agricultura',
    'Cultura Agricola Temporaria de mais de 1 Ciclo': 'agricultura',
    'Cultura Agrícula Temporaria': 'agricultura',
    'Cultura Agricola Perene': 'agricultura',
    'Cultura Agricola SemiPerene': 'agricultura',
    'Cultura Agricola Temporária 1 ciclo': 'agricultura',
    'Cultura Agricola Temporária de Mais de 1 ciclo': 'agricultura',
    'Semi-Perennial Agricultural Culture': 'agricultura',

    'Corpo DAgua': 'agua',
    "Corpos d'água": 'agua',

    'Urbanizada': 'urbano',
}

# Rotulos que significam "esta coleção não cobre este ponto".
SEM_COBERTURA = {'No-Data', 'No-info', 'Não Observado', 'Não Observada'}

FONTE = 'TerraClass {bioma}, ano base ' + ANO_BASE


def trajetorias(locais):
    """Uma consulta ao WLTS com todos os pontos, nas duas coleções do TerraClass."""
    servico = wlts.WLTS(SERVICO_WLTS)
    trajetoria = servico.tj(latitude=locais['latitude'].tolist(),
                            longitude=locais['longitude'].tolist(),
                            collections=list(COLECOES),
                            start_date=ANO_BASE, end_date=ANO_BASE)
    tj = trajetoria.df()
    tj['point_id'] = tj['point_id'].astype(int) - 1      # o WLTS numera a partir de 1
    return tj


def rerrotular(locais, tj):
    """Combina as duas coleções num rótulo por ponto; devolve os aceitos e os descartes."""
    por_ponto = {}
    for _, linha in tj.iterrows():
        if linha['class'] in SEM_COBERTURA:
            continue
        por_ponto.setdefault(int(linha['point_id']), []).append(
            (linha['collection'], linha['class']))

    aceitos, descartes = [], []
    for i in range(len(locais)):
        registros = por_ponto.get(i, [])
        bruto = '; '.join(f'{col.removeprefix("terraclass_")}: {cl}' for col, cl in registros)
        rotulos = {MAPA[cl] for _, cl in registros if cl in MAPA}
        if len(rotulos) != 1:
            descartes.append({'i': i, 'classe_antiga': locais.iloc[i]['classe'],
                              'motivo': 'divergência entre as coleções' if rotulos
                                        else ('fora das cinco classes' if registros
                                              else 'sem cobertura no TerraClass'),
                              'terraclass': bruto or '(nenhum registro)'})
            continue
        bioma = 'Amazônia' if any(c == 'terraclass_amazonia' for c, _ in registros) else 'Cerrado'
        aceitos.append({'longitude': float(locais.iloc[i]['longitude']),
                        'latitude': float(locais.iloc[i]['latitude']),
                        'classe': rotulos.pop(),
                        'fonte': FONTE.format(bioma=bioma),
                        'classe_terraclass': '; '.join(cl for _, cl in registros)})
    return pd.DataFrame(aceitos), pd.DataFrame(descartes)


def main():
    locais = gpd.read_file(ORIGEM)
    print(f'{len(locais)} locais em {ORIGEM.name}')

    tj = trajetorias(locais)
    print(f'{len(tj)} registros do WLTS ({ANO_BASE})')

    amostras, descartes = rerrotular(locais, tj)

    gdf = gpd.GeoDataFrame(
        amostras,
        geometry=gpd.points_from_xy(amostras['longitude'], amostras['latitude']),
        crs='EPSG:4326')
    gdf.to_file(DESTINO, driver='GPKG')

    print(f'\n{DESTINO.name}: {len(gdf)} pontos')
    print(gdf.groupby(['classe', 'fonte']).size().to_string())
    print(f'\n{len(descartes)} descartados:')
    print(descartes.to_string(index=False) if len(descartes) else '  (nenhum)')


if __name__ == '__main__':
    main()
