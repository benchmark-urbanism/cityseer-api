---
layout: '@src/layouts/PageLayout.astro'
---

# Datasets

The real-world datasets used in the recipes are provided for reproducibility in the repository's [examples/data](https://github.com/benchmark-urbanism/cityseer-api/tree/master/examples/data) directory. Please refer to the respective sources for the most up-to-date versions, use the datasets in accordance with the source licenses (these should be open, but always check), and be sure to cite the original authors of the data.

Dataset preprocessing is done per the [ua-dataset-madrid](https://github.com/songololo/ua-dataset-madrid) repository, which is the canonical source for this data and is intended as a base for openly reproducible urban analytics papers, workflows, and tutorials. The street network, neighbourhoods, and premises datasets are copies of that repository's latest versions and are refreshed manually when it updates; the remaining datasets are maintained directly in this repository.

## Neighbourhoods

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_nbhds/madrid_nbhds.gpkg)

[Source](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=760e5eb0d73a7710VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default) | [License](https://datos.madrid.es/egob/catalogo/aviso-legal)

Origin of the data: Madrid City Council (or, where appropriate, administrative body, body or entity in question).

Description: Delimitation of the 131 neighborhoods of the municipality of Madrid. The names and codes of each neighborhood and the districts to which they belong are indicated. The initial delimitation corresponds to the territorial restructuring of 1987.

## Boundary

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_bounds/madrid_bounds.gpkg)

Derived from the neighbourhoods dataset above; same source and license.

## Blocks

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_blocks/madrid_blocks.gpkg)

[Source](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018) | [License](https://land.copernicus.eu/en/data-policy)

The Copernicus land monitoring products and services are made available on a principle of full, open and free access, as established by the Commission Delegated Regulation (EU) No 1159/2013 of 12 July 2013, on the conditions that:

1. When distributing or communicating Copernicus Land Monitoring Service products and services (data, software scripts, web services, user and methodological documentation and similar) to the public, users shall inform the public of the source of these products and services.
2. Where the Copernicus Land Monitoring Service products and services have been adapted or modified by the user, the user shall clearly state this.
3. Users shall make sure not to convey the impression to the public that the user's activities are officially endorsed by the European Union.

Urban Atlas Land Cover/Land Use 2018 (vector), Europe, 6-yearly. European Union's Copernicus Land Monitoring Service information, [https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018). [https://doi.org/10.2909/fb4dffa1-6ceb-4cc0-8372-1ed354c285e6](https://doi.org/10.2909/fb4dffa1-6ceb-4cc0-8372-1ed354c285e6)

## Premises

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_premises/madrid_premises.gpkg)

[Source](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=66665cde99be2410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default) | [License](https://datos.madrid.es/egob/catalogo/aviso-legal)

Origin of the data: Madrid City Council (or, where appropriate, administrative body, body or entity in question).

Description: Microdata file of the census of premises and activities of the Madrid City Council, classified according to their type of access (street door or grouped), situation (open, closed...) and indication of the economic activity exercised and the hospitality and restaurant terraces that appear registered in said census.

Preprocessing performed per: [ua-dataset-madrid](https://github.com/songololo/ua-dataset-madrid).

Key columns:

| Column              | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `local_distr_desc`  | District name                                                  |
| `local_neighb_desc` | Neighbourhood name                                             |
| `section_id`        | Activity section code (e.g., I = hospitality, R = recreation)  |
| `section_desc`      | Activity section description                                   |
| `division_id`       | Activity division code                                         |
| `division_desc`     | Activity division description                                  |
| `epigraph_id`       | Specific activity code                                         |
| `epigraph_desc`     | Specific activity description (e.g., RESTAURANTE)              |

## Overture Maps Buildings

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_buildings/madrid_bldgs.gpkg)

[Source](https://docs.overturemaps.org/guides/buildings/) | [License](https://docs.overturemaps.org/attribution/)

License for theme: ODbL

- © OpenStreetMap contributors. Available under the Open Database License.
- Esri Community Maps contributors. Available under CC BY 4.0.
- Global ML Building Footprints. Licensed by Microsoft under the Open Database License.
- Google Open Buildings. Available under CC BY 4.0.
- USGS 3D Elevation Program Digital Elevation Program.
- Qian Shi, et al. A First High-quality Vector Data of Buildings in East Asian Countries Based on a Comprehensive Large-scale Mapping Framework. Zenodo, 22 July 2023, doi:10.5281/zenodo.8174931. Available under CC BY 4.0.

Key columns:

| Column              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `mean_height`       | Average building height in metres (where available)         |
| `area`              | Footprint area in square metres                             |
| `perimeter`         | Footprint perimeter in metres                               |
| `compactness`       | Compactness ratio (area relative to perimeter)              |
| `orientation`       | Orientation angle in degrees                                |
| `volume`            | Estimated building volume (height x area, where available)  |
| `floor_area_ratio`  | Floor area ratio                                            |
| `form_factor`       | Building form factor                                        |
| `corners`           | Number of corners in the footprint                          |
| `shape_index`       | Shape index metric                                          |
| `fractal_dimension` | Fractal dimension of the footprint boundary                 |

## Overture Maps Infrastructure

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_infrastucture/madrid_infrast.gpkg)

[Source](https://docs.overturemaps.org/guides/base/) | [License](https://docs.overturemaps.org/attribution/)

License for theme: ODbL

- © OpenStreetMap contributors. Available under the Open Database License.
- Data from the Daylight Map Distribution
- ESA WorldCover. Available under CC BY 4.0 DEED.
- Data products from ETOPO1. Available under Open Data Commons Public Domain Dedication and License.
- Data from GLOBathy. Available under CC0 1.0 (assumed).

## Streets

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_streets/street_network.gpkg) | [3D GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_streets/street_network_3d.gpkg)

[Source](https://centrodedescargas.cnig.es/CentroDescargas/index.jsp) (CNIG/IGN - Redes de Transporte - Madrid province) | [License](https://creativecommons.org/licenses/by/4.0/legalcode.es) ([ign.es](https://www.ign.es))

Cite: Attribute IGN (Instituto Geográfico Nacional de España).

Description: Road network (Red Viaria) for Madrid province from the CNIG/IGN Redes de Transporte dataset. Layer `rt_tramo_vial` containing road segments with classification, surface, lane, and naming attributes.

Preprocessing: Download `RT_MADRID_gpkg.zip` from CNIG Centro de Descargas. Extract `red_viaria.gpkg`, layer `rt_tramo_vial`. Reproject from EPSG:4258 to EPSG:25830. Clip to 20km buffered bounds. Retain `clased` (road class) and `nombre` (road name) attributes only. Set coordinate grid precision to 1m and simplify geometries to 1m tolerance. Further preprocessing performed per [ua-dataset-madrid](https://github.com/songololo/ua-dataset-madrid).

Key columns:

| Column   | Description                        |
| -------- | ---------------------------------- |
| `clased` | Road classification (e.g., Urbano) |
| `nombre` | Street name                        |

## Census

[GeoPackage](https://github.com/benchmark-urbanism/cityseer-api/blob/master/examples/data/madrid_census/eu_stat_clipped.gpkg)

[Source](https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/geostat) | [License](https://ec.europa.eu/eurostat/web/main/help/copyright-notice)

Copyright European Union 2025. The source needs to be indicated and when re-use involves modifications to the data or text, this must be stated clearly to the end user of the information.

| Code    | Description                          |
| ------- | ------------------------------------ |
| T       | Total population                     |
| M       | Male population                      |
| F       | Female population                    |
| Y_LT15  | Age under 15 years                   |
| Y_1564  | Age 15 to 64 years                   |
| Y_GE65  | Age 65+ years                        |
| EMP     | Employed persons                     |
| NAT     | Born in reporting country            |
| EU_OTH  | Born in other EU Member State        |
| OTH     | Born elsewhere                       |
| SAME    | Residence unchanged in past year     |
| CHG_IN  | Moved within reporting country       |
| CHG_OUT | Moved from outside reporting country |

## GTFS - Metro

[GTFS files](https://github.com/benchmark-urbanism/cityseer-api/tree/master/examples/data/madrid_gtfs/madrid_metro) (stops, stop times, routes, trips, shapes, frequencies)

[Source](https://mobilitydatabase.org/feeds/gtfs/mdb-794) | [License](https://mobilitydatabase.org/terms-and-conditions)
