# %%
# runs example local analysis for specified streets
# hybrid API: fine network decomposition (graphs.nx_decompose) is a tools-level
# preprocessing step with no CityNetwork equivalent; everything downstream (dual
# construction, live boundary, centrality, accessibilities) uses CityNetwork
# see https://cityseer.benchmarkurbanism.com/guide/fundamentals#citynetwork-api
import json

import pandas as pd
from cityseer.network import CityNetwork
from cityseer.tools import graphs, io
from osmnx import _errors as ox_errs
from osmnx import features
from pyproj import Transformer
from shapely import geometry


def local_analysis(line_geom: geometry.LineString, location_key: str) -> None:
    """ """
    print(f"Processing {location_key}")
    # build the network
    # buffer by 5000 for centrality
    extent = line_geom.buffer(50)
    extent_buff = extent.buffer(5000)
    G_multigraph = io.osm_graph_from_poly(extent_buff, poly_crs_code=27700, to_crs_code=27700, simplify=True)
    # decomposition is the one tools-level step CityNetwork does not expose
    G_decomp = graphs.nx_decompose(G_multigraph, 10)
    # CityNetwork builds the dual graph automatically; the boundary marks nodes inside
    # the extent as live. remove_fillers=False keeps the decomposed degree-2 nodes,
    # which the default filler-welding would otherwise undo
    cn = CityNetwork.from_nx(G_decomp, boundary=extent, remove_fillers=False)
    cn = cn.centrality_shortest(distances=[500, 1000, 2500, 5000])
    cn = cn.centrality_simplest(distances=[500, 1000, 2500, 5000])

    # a sample landuse schema
    SCHEMA = {
        "drinking": {"amenity": ["bar", "biergarten", "pub"]},
        "beverage": {"amenity": ["cafe"]},
        "eating": {"amenity": ["fast_food", "food_court", "ice_cream", "restaurant"]},
        "children": {
            "amenity": ["kindergarten", "childcare"],
            "leisure": ["playground"],
        },
        "education": {
            "amenity": [
                "school",
                "college",
                "language_school",
                "research_institute",
                "training",
                "university",
                "music_school",
            ]
        },
        "transport": {
            "railway": ["subway", "station"],
            "highway": ["bus_stop"],
        },
        "grocery": {
            "shop": [
                "alcohol",
                "bakery",
                "beverages",
                "brewing_supplies",
                "butcher",
                "cheese",
                "chocolate",
                "coffee",
                "confectionery",
                "convenience",
                "deli",
                "dairy",
                "farm",
                "frozen_food",
                "general",
                "greengrocer",
                "health_food",
                "ice_cream",
                "pasta",
                "pastry",
                "seafood",
                "spices",
                "supermarket",
                "tea",
                "wine",
            ]
        },
        "retail": {
            "shop": [
                "department_store",
                "florist",
                "kiosk",
                "mall",
                "wholesale",
                "baby_goods",
                "bag",
                "boutique",
                "clothes",
                "fabric",
                "fashion_accessories",
                "jewelry",
                "leather",
                "sewing",
                "shoes",
                "tailor",
                "watches",
                "wool",
                "beauty",
                "chemist",
                "cosmetics",
                "hairdresser",
                "hairdresser_supply",
                "hearing_aids",
                "herbalist",
                "massage",
                "medical_supply",
                "nutrition_supplements",
                "optician",
                "perfumery",
                "tattoo",
                "agrarian",
                "appliance",
                "bathroom_furnishing",
                "doityourself",
                "electrical",
                "energy",
                "fireplace",
                "garden_centre",
                "garden_furniture",
                "glaziery",
                "groundskeeping",
                "hardware",
                "houseware",
                "locksmith",
                "paint",
                "pottery",
                "security",
                "trade",
                "antiques",
                "bed",
                "candles",
                "carpet",
                "curtain",
                "doors",
                "flooring",
                "furniture",
                "household_linen",
                "interior_decoration",
                "kitchen",
                "lighting",
                "tiles",
                "window_blind",
                "computer",
                "electronics",
                "hifi",
                "mobile_phone",
                "radiotechnics",
                "telecommunication",
                "vacuum_cleaner",
                "art",
                "camera",
                "collector",
                "craft",
                "frame",
                "games",
                "model",
                "music",
                "photo",
                "trophy",
                "video",
                "video_games",
                "anime",
                "books",
                "gift",
                "lottery",
                "newsagent",
                "stationery",
                "ticket",
                "copyshop",
                "dry_cleaning",
                "laundry",
                "pawnbroker",
                "pet",
                "toys",
                "travel_agency",
            ],
            "amenity": ["internet_cafe"],
        },
    }

    # get the landuses data
    # cast polygon to 4326
    transformer = Transformer.from_crs(27700, 4326, always_xy=True)

    def transform_polygon(polygon, transformer):
        return geometry.Polygon([transformer.transform(x, y) for x, y in polygon.exterior.coords])

    # use a 2km buffer for landuses
    transformed_polygon_4326 = transform_polygon(extent.buffer(2000), transformer)
    # fetch data
    dfs = []
    for cat_key, val in SCHEMA.items():
        print(f"Retrieving {cat_key}")
        # iterate the nested OSM keys
        for osm_key, osm_vals in val.items():
            # fetch OSM data for each key and value pairing
            try:
                data_gdf = features.features_from_polygon(transformed_polygon_4326, tags={osm_key: osm_vals})
            except ox_errs.InsufficientResponseError as e:
                print(e)
                continue
            # extract centroids from ways
            data_gdf = data_gdf.to_crs(27700)
            data_gdf.loc[:, "centroids"] = data_gdf.geometry.centroid
            data_gdf.drop(columns=["geometry"], inplace=True)
            data_gdf.rename(columns={"centroids": "geom"}, inplace=True)
            data_gdf.set_geometry("geom", inplace=True)
            data_gdf.set_crs(27700, inplace=True)
            # reset the index
            data_gdf = data_gdf.reset_index(level=0, drop=True)
            data_gdf.index = data_gdf.index.astype(str)
            # add the category and OSM keys to the GDF
            data_gdf["cat_key"] = cat_key
            data_gdf["osm_key"] = osm_key
            # add the values as JSON
            data_gdf["osm_vals"] = json.dumps(osm_vals)
            # reduce the GDF to only the wanted columns
            data_gdf = data_gdf[["cat_key", "osm_key", "osm_vals", "geom"]]
            # append to GDF list
            dfs.append(data_gdf)
    # concatenate the list of GDF into a single GDF
    landuses_gdf = pd.concat(dfs)
    # reset the index
    landuses_gdf = landuses_gdf.reset_index()
    landuses_gdf.index = landuses_gdf.index.astype(str)
    landuses_gdf = landuses_gdf.to_crs(27700)
    landuses_gdf = landuses_gdf.reset_index(level=0, drop=True)
    landuses_gdf.index = landuses_gdf.index.astype(str)
    landuses_gdf = landuses_gdf[["cat_key", "geom"]]
    landuses_gdf.to_file(f"../temp/{location_key}_places.gpkg")
    # compute accessibilities
    cn = cn.compute_accessibilities(
        landuses_gdf,
        landuse_column_label="cat_key",
        accessibility_keys=list(SCHEMA.keys()),
        distances=[100, 200, 500],
    )
    #
    nodes_gdf = cn.to_geopandas()
    nodes_gdf.loc[nodes_gdf.live].to_file(f"../temp/{location_key}.gpkg")


# %%
oxford_street_geom = geometry.LineString([[528680, 181151], [528983, 181222]])
local_analysis(oxford_street_geom, "oxford_street")
hackney_walk_geom = geometry.LineString([[535225, 184951], [535354, 184952]])
local_analysis(hackney_walk_geom, "hackney_walk")
chats_geom = geometry.LineString([[535624, 185731], [535719, 185457]])
local_analysis(chats_geom, "chatsworth_road")
olympic_geom = geometry.LineString([[537870, 184912], [538170, 185185]])
local_analysis(olympic_geom, "olympic_village")
howland_geom = geometry.LineString([[529204, 181854], [529421, 182014]])
local_analysis(howland_geom, "howland road")
goodge_geom = geometry.LineString([[529356, 181640], [529573, 181784]])
local_analysis(goodge_geom, "goodge_street")
