# Estudo de caso

Notebook que exercita o cliente `wtss.py` reformulado numa análise completa, do acesso
aos dados à interpretação do resultado. É autocontido e reprodutível: basta rede e as
dependências abaixo.

| Notebook | O que demonstra |
| --- | --- |
| [`caso1_ndvi_temporal.ipynb`](caso1_ndvi_temporal.ipynb) | NDVI ao longo do tempo numa região agrícola: consulta multi-banda, metadados das bandas (`scale`, `nodata` e o deslocamento declarado em `scale_add`), álgebra entre bandas (`B8A`/`B04`), aplicação opcional de `scale`/`offset`, qualidade pela banda `SCL`, fatiamento por ano agrícola, passagem do pixel para a área (`summarize`, `plot_map`), as estruturas esparsa e densa devolvidas pela consulta por `geom` (o cubo denso na grade nativa da coleção), a comparação do mapa do cliente com o COG original lido por `rasterio`, a exportação em GeoTIFF com `to_geotiff()` conferida por releitura, gráficos interativos (`plotly` e `folium`), a conferência dos rótulos com as coleções oficiais pelo WLTS e a classificação de uso do solo com a série temporal como atributo |

## Área de estudo

O ponto analisado é **−13.55, −56.00 (Nova Mutum, MT)**, área de soja de verão seguida
de milho safrinha, na coleção `S2-16D-2` (Sentinel-2, 10 m).

O período é **2022-09-01 a 2025-08-31**, alinhado ao calendário agrícola de Mato Grosso,
que vai de setembro a agosto. São três anos agrícolas completos — 2022/23, 2023/24 e
2024/25 — cada um com as 23 composições de 16 dias que o cubo produz no intervalo. O
período para nessas três safras por causa dos rótulos da classificação, do ano base 2020:
cada safra a mais afastaria ainda mais as séries do ano dos rótulos (seção 1 do notebook).
A safra 2024/25, a última do período, é a que a seção 9 recorta em área e o capítulo 10
classifica.

## Amostras rotuladas

O caso 1 classifica uso do solo em cinco classes com `amostras_terraclass_5classes.gpkg`,
gerado por `preparar_amostras_terraclass.py`. Os rótulos vêm do **TerraClass**
(INPE/Embrapa), mapeamento oficial de 30 m derivado de imagem, consultado pelo
[WLTS](https://data.inpe.br/bdc/wlts/v1/) no ano base **2020** — o mais recente em que as
duas folhas da coleção devolvem classe. A folha do Cerrado declara dados até 2022, mas não
devolve classe para esse ano nas amostras que a seção 10.1 consulta.

A região fica na fronteira dos biomas, então nenhuma folha cobre sozinha todos os
pontos: `terraclass_amazonia` devolve `No-Data` fora da Amazônia Legal e
`terraclass_cerrado` devolve `No-info` fora do Cerrado. As duas são consultadas e
combinadas, e a coluna `fonte` registra de qual delas veio cada rótulo. As classes
originais são reagrupadas nas cinco do caso — vegetação natural florestal → `floresta`,
pastagem herbácea/arbustiva → `pastagem`, culturas agrícolas → `agricultura`, corpo
d'água → `agua`, urbanizada → `urbano` — e o que não cai em nenhuma delas (silvicultura,
mineração, outros usos) é descartado.

Os **locais** são os mesmos de `amostras_5classes.gpkg`, o conjunto anterior, que
continua no repositório: o que mudou foi a fonte do rótulo, não a amostragem. Esse
arquivo anterior é usado na seção 10.2 do notebook, que cruza os seus rótulos com as
coleções oficiais pelo WLTS e é o que justifica a troca. Ele foi montado por
`preparar_amostras_5classes.py`, com `floresta`, `pastagem` e `agricultura` recortadas de
`samples_matogrosso_mod13q1` (pacote [`sits`](https://github.com/e-sensing/sits),
INPE/e-sensing, guardado aqui como `amostras_uso_do_solo_mt.gpkg`, rótulos de 2000/01 a
2015/16) e `agua` e `urbano` sorteadas sobre a Coleção 9 do MapBiomas, cobertura 2023.

## Arquivos da pasta

| Arquivo | O que é |
| --- | --- |
| `caso1_ndvi_temporal.ipynb` | o notebook, com as saídas da última execução |
| `caso1_ndvi_temporal.html` | o notebook exportado por `nbconvert`, com os gráficos interativos; é gerado a partir do notebook e não vem no repositório |
| `amostras_terraclass_5classes.gpkg` | as 218 amostras rotuladas pelo TerraClass, usadas no capítulo 10; gerado por `preparar_amostras_terraclass.py` |
| `amostras_5classes.gpkg` | as 223 amostras do conjunto anterior (`sits` + MapBiomas), usadas na seção 10.2; gerado por `preparar_amostras_5classes.py` |
| `amostras_uso_do_solo_mt.gpkg` | cópia de `samples_matogrosso_mod13q1`, do `sits`, de onde `preparar_amostras_5classes.py` recorta `floresta`, `pastagem` e `agricultura` |
| `recorte_cog_ndvi_20250407.tif` | janela da área lida do COG original do BDC por `preparar_recorte_cog.py`; alimenta a figura tripla da seção 9.4 sem depender de rede, e a seção 9.5 confere contra ela o GeoTIFF exportado |
| `area_ano_agricola_2024_25.tif` | saída de `to_geotiff()` na seção 9.5, gravada pela execução do notebook e que não vem no repositório: o NDVI de 2024/25, uma banda por data (23), 43×46 pixels de 10 m no CRS nativo da coleção (a Albers do BDC), e não em lon/lat; a seção 9.5 o relê e o confere contra o cubo em memória e o recorte do COG |
| `preparar_amostras_terraclass.py` | gera `amostras_terraclass_5classes.gpkg` a partir dos locais de `amostras_5classes.gpkg` e do TerraClass, pelo WLTS |
| `preparar_amostras_5classes.py` | gera `amostras_5classes.gpkg` a partir do `sits` e do MapBiomas |
| `preparar_recorte_cog.py` | gera `recorte_cog_ndvi_20250407.tif` |

## Dependências

```bash
pip install -e ".[examples,dev]"
```

O notebook usa `pandas`, `numpy`, `matplotlib`, `seaborn`, `geopandas`, `rasterio`,
`pyproj`, `scikit-learn`, `xarray`, `plotly`, `folium` e `wlts`. O `pyproj` é dependência
direta do pacote; as demais vêm dos extras `examples`, `io`, `xarray` e `interactive` do
`pyproject.toml`. As figuras interativas da seção 9.6 e a exportação em GeoTIFF dependem
dos extras `interactive` e `io`; as seções 10.1 e 10.2 dependem do `wlts`.

### Leitura remota de COG no Windows

A célula opcional da seção 9.4 lê o COG direto do servidor. O `rasterio` aponta
`GDAL_CURL_CA_BUNDLE` para o `cacert.pem` do `certifi`; se esse caminho tiver acento
(`Área de Trabalho`, por exemplo), o schannel não consegue abri-lo e o GDAL falha com
`CURL error: schannel: invalid path name for CA file`. A saída é copiar o `cacert.pem`
para um caminho sem acentos e apontar `GDAL_CURL_CA_BUNDLE` para lá — o notebook e
`preparar_recorte_cog.py` fazem isso sozinhos. A figura da seção 9.4 não depende dessa
célula: ela lê o recorte local.

## Servidor

As consultas apontam para `https://data.inpe.br/bdc/wtss/v4/`. Coleções autenticadas do
Brazil Data Cube exigem um token, passado na construção do serviço:

```python
service = WTSS('https://data.inpe.br/bdc/wtss/v4/', access_token='SEU_TOKEN')
```

## Reexecução

```bash
jupyter nbconvert --to notebook --execute --inplace examples/caso1/caso1_ndvi_temporal.ipynb
```

O kernel precisa ter o pacote e os extras acima instalados. Se o kernel registrado no
notebook apontar para outro ambiente, indique o certo com
`--ExecutePreprocessor.kernel_name=<kernel>`.

As seções 10.1 e 10.2 consultam o WLTS, que limita requisições: em consultas seguidas ele
passa a responder 429 (*Too Many Requests*). O notebook repete essas consultas com espera
crescente, de modo que reexecuções em sequência podem levar vários minutos a mais nessas
seções; se ainda assim uma delas falhar, basta repeti-la depois de alguns minutos.

O notebook está versionado **com as saídas** para servirem de evidência de execução.
Os valores numéricos citados no texto correspondem às séries recuperadas na data da
última execução; como as coleções continuam sendo alimentadas, uma reexecução com
período estendido pode alterá-los.
