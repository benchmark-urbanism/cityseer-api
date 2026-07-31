import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GeoPandas

    [GeoPandas](https://geopandas.org/) extends the pandas data analysis library to work with spatial data. Its core data structure, the `GeoDataFrame`, is a table in which each row is a feature: one column holds a Shapely geometry, and the remaining columns hold attributes. A GeoDataFrame also records its coordinate reference system (CRS).

    To see how the pieces fit together, we will build a small GeoDataFrame from scratch.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    from shapely.geometry import Point

    return Point, gpd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, we'll define some data as a list of dictionaries. Each dictionary will represent a city, storing its name and approximate coordinates (longitude, latitude).
    """)
    return


@app.cell
def _():
    data = [
        {"city": "Seville", "x": -5.9845, "y": 37.3891},
        {"city": "Barcelona", "x": 2.1734, "y": 41.3851},
        {"city": "Valencia", "x": -0.3774, "y": 39.4699},
    ]

    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we need to create the corresponding geometry objects. As these are coordinates, `Point` objects are suitable. We can generate a list of `Point` objects directly from our data.
    """)
    return


@app.cell
def _(Point, data):
    # Create Point geometries from the x and y columns
    locations = []

    for city in data:
        x = city["x"]
        y = city["y"]
        geom = Point(x, y)
        locations.append(geom)

    locations
    return (locations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the structured data and geometry prepared, the final component is the CRS. Given that the coordinates are longitude and latitude, the appropriate CRS is WGS 84, identified by the EPSG code 4326.

    We can now construct the `GeoDataFrame`:
    """)
    return


@app.cell
def _(data, gpd, locations):
    gdf_cities = gpd.GeoDataFrame(
        data,  # The list of dictionaries
        geometry=locations,  # The Shapely Points
        crs=4326,  # Coordinate Reference System (WGS 84)
    )

    gdf_cities
    return (gdf_cities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Indexing

    Similar to pandas DataFrames, GeoDataFrames possess an index. By default, this is a range of integers. You can assign a more meaningful index using one of the columns (e.g., the city name) with the `set_index()` method. This often simplifies data selection.
    """)
    return


@app.cell
def _(gdf_cities):
    gdf_cities_idx = gdf_cities.set_index("city")
    gdf_cities_idx
    return (gdf_cities_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can access a row using its index. Pandas provides a special indexing method, `.loc[]`, which allows you to access rows by their index label. This is particularly useful when you've set a meaningful index, such as city names.
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.loc["Seville"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also specify a column name. For instance, to retrieve the latitude of Barcelona:
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.loc["Barcelona", "y"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To retrieve all rows for a specific column, use the `:` operator. For example, to get all rows for the `y` column:
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.loc[:, "y"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An alternative way to access columns is by using the column name directly:
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx["y"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GeoDataFrames can filter rows based on their properties. To do this, you create a boolean mask, which can then be used to exclude rows you are not interested in. For instance, if you want to filter the cities to include only those with a latitude greater than 40, you can proceed as follows:
    """)
    return


@app.cell
def _(gdf_cities_idx):
    # Create a boolean mask
    mask = gdf_cities_idx["y"] > 40
    mask
    return (mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This mask is a pandas Series of boolean values, where `True` indicates that the condition (y > 40) is met, and `False` indicates it is not. You can now apply this mask to filter the rows in the original GeoDataFrame.
    """)
    return


@app.cell
def _(gdf_cities_idx, mask):
    # Use the mask to filter the rows
    gdf_cities_idx.loc[mask]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As shown, only two rows are returned because Valencia, the third city, has a latitude below 40.

    ## Geospatial

    GeoDataFrames possess special properties stemming from their spatial nature.

    GeoPandas formally designates a particular column as the geometry; this is typically done during the creation of the GeoDataFrame. The geometry column contains Shapely geometry objects, which can be points, lines, or polygons. This column is often called `geometry`, but this is not always the case.
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.geometry
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, GeoPandas formally associates the geometry with a given CRS, accessible via the `.crs` attribute.
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.crs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GeoPandas conveniently integrates with `matplotlib` for straightforward plotting. Calling `.plot()` on a GeoDataFrame will render its geometries.
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_idx.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coordinate Transformations

    > **Note:**
    > #### Geographic vs. projected CRS
    >
    > A **geographic CRS** (e.g., WGS 84 / EPSG:4326) stores coordinates as longitude and latitude in **degrees** on the Earth's curved surface. A **projected CRS** (e.g., EPSG:3035, UTM zones) projects coordinates onto a flat surface in **metres**.
    >
    > You generally want to use a projected CRS for the type of work we're doing in this notebook. To find an appropriate projected CRS for your study area, search for your city or country at [epsg.io](https://epsg.io). Common choices include UTM zones (accurate for local areas), EPSG:3035 (Europe-wide), and national projections like EPSG:27700 (Great Britain).

    GeoPandas makes CRS transformation very easy with the `to_crs()` method. Let's convert our cities GeoDataFrame to EPSG:3035, a projected equal-area CRS suitable for Europe (ETRS89-LAEA).
    """)
    return


@app.cell
def _(gdf_cities_idx):
    gdf_cities_proj = gdf_cities_idx.to_crs(3035)
    gdf_cities_proj
    return (gdf_cities_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the `x` and `y` columns no longer match the geometry, as the geometry is now in metres according to EPSG:3035. The `.crs` attribute reflects this change:
    """)
    return


@app.cell
def _(gdf_cities_proj):
    gdf_cities_proj.crs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting the transformed data shows the locations relative to the new projected coordinate system.
    """)
    return


@app.cell
def _(gdf_cities_proj):
    gdf_cities_proj.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spatial Operations

    GeoPandas enables you to apply spatial operations directly to `GeoDataFrames` or `GeoSeries`. These operations are applied element-wise.

    For instance, to create a 100km buffer around each city:
    """)
    return


@app.cell
def _(gdf_cities_proj):
    # The buffer distance is in the units of the CRS (metres for EPSG:3035)
    gdf_cities_proj["geometry"] = gdf_cities_proj.geometry.buffer(100000)  # 100km buffer

    gdf_cities_proj.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, you can calculate the distance from each feature in a GeoDataFrame to a single Shapely geometry object. Let's determine the distance from each city buffer to a specific point (also defined in EPSG:3035 coordinates):
    """)
    return


@app.cell
def _(Point, gdf_cities_proj):
    # Define a point in the same CRS (EPSG:3035)
    pt = Point(3300000, 2000000)

    # Calculate distance from each geometry in the GeoSeries to the point
    gdf_cities_proj.geometry.distance(pt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating and Updating Columns

    If we wished to create a new column named 'easting' and assign it the x-coordinate of the geometry, we could proceed as follows:
    """)
    return


@app.cell
def _(gdf_cities_proj):
    gdf_cities_proj["easting"] = gdf_cities_proj.geometry.centroid.x

    gdf_cities_proj
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each geometry row in the DataFrame is a Shapely geometry, so you can use standard Shapely syntax to access properties such as the centroid or its x-coordinate.

    If you decide instead to update the existing `x` and `y` columns, you can reference them directly:
    """)
    return


@app.cell
def _(gdf_cities_proj):
    gdf_cities_proj["x"] = gdf_cities_proj.geometry.centroid.x
    gdf_cities_proj["y"] = gdf_cities_proj.geometry.centroid.y

    gdf_cities_proj
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DataFrames offer an extensive variety of useful methods; here, we will use the `drop` method to remove the column we no longer wish to retain:
    """)
    return


@app.cell
def _(gdf_cities_proj):
    gdf_cities_proj.drop(columns=["easting"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Working with Data

    ### Reading Data

    GeoPandas can read various spatial data formats, such as Shapefiles, GeoPackages, and GeoJSON files. The `read_file()` function is the primary method for this. The path below is relative to this notebook's location (`examples/class/`); the bundled datasets live in `examples/data/`.
    """)
    return


@app.cell
def _(gpd):
    mad_bldgs = gpd.read_file("../data/madrid_buildings/madrid_bldgs.gpkg")

    mad_bldgs.head()
    return (mad_bldgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `head()` method shows the first few rows of the GeoDataFrame, including the geometry column. The CRS is automatically detected and set.

    ### Plotting

    Let's plot the data.
    """)
    return


@app.cell
def _(mad_bldgs):
    mad_bldgs.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To zoom in when plotting, you can set your x and y-axis limits. For a cleaner plot, it is also generally preferable to turn off the axes so that the coordinates do not render.
    """)
    return


@app.cell
def _(mad_bldgs):
    _ax = mad_bldgs.plot()
    _ax.set_xlim(439000, 442000)
    _ax.set_ylim(4473000, 4476000)
    _ax.axis("off")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create a new column named `area`, which we will set to the value of each geometry's area. Then, we will plot the data again, this time rendering the colour according to the building's area.
    """)
    return


@app.cell
def _(mad_bldgs):
    mad_bldgs["area"] = mad_bldgs.geometry.area
    _ax = mad_bldgs.plot(
        column="area",
        cmap="viridis",
        vmax=10000,
        legend=True,
        legend_kwds={"label": "Building footprint area (m²)", "shrink": 0.6},
    )
    _ax.set_title("Buildings coloured by footprint area")
    _ax.set_xlim(439000, 442000)
    _ax.set_ylim(4473000, 4476000)
    _ax.axis("off")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving Data

    GeoPandas can save GeoDataFrames to various formats, including Shapefiles, GeoPackages, and GeoJSON files. The `to_file()` method is used for this purpose. You can specify the format using the `driver` parameter.
    """)
    return


@app.cell
def _(mad_bldgs):
    mad_bldgs.to_file("bldgs_w_area.gpkg", driver="GPKG")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercises

    - Load the bundled buildings GeoPackage and filter to only buildings with an area greater than 500 square metres. How many remain?
    - Reproject the buildings GeoDataFrame to EPSG:4326 (WGS 84) and plot the result. What changes?
    - Create a new column called `perimeter` containing each building's perimeter length. Plot the buildings coloured by perimeter.
    - Create a `Point` geometry for a location of your choice and calculate the distance from each building's centroid to that point. Which building is closest?

    ## Summary

    - GeoPandas extends pandas with a geometry column and CRS support, bridging tabular data analysis with spatial operations.
    - Use `.loc[]` and boolean masks to index and filter rows by attribute or spatial properties.
    - `to_crs()` transforms between coordinate reference systems; always use a projected CRS for distance and area calculations.
    - Spatial operations like `buffer` and `distance` work element-wise across the GeoDataFrame.
    - `read_file()` and `to_file()` handle common spatial formats (GeoPackage, Shapefile, GeoJSON).

    Next: [Urban Analytics](https://cityseer.benchmarkurbanism.com/start/5-urban) introduces `osmnx` for downloading OSM data and `momepy` for morphological analysis. The skills from this chapter are used extensively in the [Network Preparation](https://cityseer.benchmarkurbanism.com/examples/networks) and [Statistics](https://cityseer.benchmarkurbanism.com/examples/stats) recipes.
    """)
    return


if __name__ == "__main__":
    app.run()
