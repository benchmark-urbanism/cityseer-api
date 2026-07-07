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
    # Spatial Data (Shapely)

    [Shapely](https://shapely.readthedocs.io/) is a Python package for creating and manipulating geometric objects such as points, lines, and polygons. It underpins much of the Python geospatial ecosystem, including GeoPandas, which we cover in the next lesson.

    Points represent a single location, defined by x and y coordinates:
    """)
    return


@app.cell
def _():
    from shapely.geometry import LineString, Point, Polygon

    pt = Point(0, 0)
    pt
    return LineString, Point, Polygon, pt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Linestrings represent sequences of connected points. You can define them using a series of XY coordinate pairs or a list of Point objects.
    """)
    return


@app.cell
def _(LineString):
    line = LineString([(0, 5), (5, 0), (10, 5)])

    line
    return (line,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polygons are closed shapes defined by a sequence of coordinates. Similar to linestrings, you can create them from a list of points or coordinate tuples.
    """)
    return


@app.cell
def _(Polygon):
    poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    poly
    return (poly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Shapely geometries come with useful properties that provide common information. For instance, you can easily retrieve a polygon's area:
    """)
    return


@app.cell
def _(poly):
    poly.area
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, you can get the length of a linestring:
    """)
    return


@app.cell
def _(line):
    line.length
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other handy properties include centroids, x and y coordinates, and WKT (Well-Known Text) representations:
    """)
    return


@app.cell
def _(poly):
    poly.centroid
    return


@app.cell
def _(pt):
    pt.x, pt.y
    return


@app.cell
def _(line):
    line.wkt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Operations and Predicates

    Shapely supports a range of typical GIS spatial operations and predicates. For example, you can measure the distance between two spatial objects:
    """)
    return


@app.cell
def _(line, pt):
    pt.distance(line)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Crucially, Shapely operates using Cartesian coordinates. For accurate distance calculations or predicate operations, ensure your geometries are in a projected coordinate reference system and that they share the same system.

    The Shapely documentation provides comprehensive details on all available operations and predicates. For instance, the [`geometry.Point`](https://shapely.readthedocs.io/en/stable/reference/shapely.Point.html#shapely.Point) page describes everything related to point geometries.
    """)
    return


@app.cell
def _(pt):
    pt_buff = pt.buffer(2)

    pt_buff
    return (pt_buff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As an example, the documentation shows that the `Point` geometry has a `buffer` method. This method accepts a distance parameter and returns a polygon representing the buffered area.

    Common spatial predicates are also readily available. For example, you can use the `within` method to check if a point is located inside a polygon:
    """)
    return


@app.cell
def _(poly, pt):
    pt.within(poly)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If a point is on a polygon's boundary, `within` returns `False`, but `intersects` or `touches` return `True`.
    """)
    return


@app.cell
def _(poly, pt):
    pt.intersects(poly)
    return


@app.cell
def _(poly, pt):
    pt.touches(poly)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Buffers, Unions, and Differencing

    Other common geometric operations include `buffer`. For instance, let's create a buffer around a linestring:
    """)
    return


@app.cell
def _(line):
    line_buff = line.buffer(2)

    line_buff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can use the `union` method to combine two polygons into a single, unified geometry:
    """)
    return


@app.cell
def _(Polygon, poly):
    poly2 = Polygon([(5, 5), (5, 15), (15, 15), (15, 5)])

    union_poly = poly.union(poly2)

    union_poly
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Operations like `difference` also behave intuitively, as you might expect from GIS software:
    """)
    return


@app.cell
def _(poly, pt_buff):
    diff_poly = poly.difference(pt_buff)

    diff_poly
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some operations offer configurable parameters for finer control. For example, the `buffer` method includes `cap_style` and `join_style` parameters, which control the appearance of the buffer's ends and joins around a line. The default is `"round"`; compare the flat-ended version below with the rounded buffer above.
    """)
    return


@app.cell
def _(line):
    buffer_flat = line.buffer(2, cap_style="flat", join_style="mitre")

    buffer_flat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Workflows

    Shapely's spatial operations and predicates closely mirror those found in GIS software like QGIS. This means tasks you might perform in a graphical user interface (UI) can often be replicated in Python, and vice-versa. Python becomes particularly advantageous for complex or lengthy workflows, as scripting enables advanced processing that can be cumbersome or difficult to achieve through a UI.

    Let's consider a simple example: modelling streets and checking if they are within a specified distance threshold of particular land uses.
    """)
    return


@app.cell
def _(LineString):
    street = LineString([(0, 0), (10, 0)])
    street_buffer = street.buffer(5)

    street_buffer
    return street, street_buffer


@app.cell
def _(Point):
    land_use = Point(3, 4)
    land_use
    return (land_use,)


@app.cell
def _(land_use, street_buffer):
    is_within = land_use.within(street_buffer)
    is_within
    return


@app.cell
def _(land_use, street):
    distance_to_street = land_use.distance(street)
    distance_to_street
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In real-world scenarios, GeoPandas is frequently used for such workflows. It excels at handling multiple features, managing coordinate reference systems, performing file input/output (I/O), and plotting.

    ### Exercises

    - Create two points and calculate the distance between them.
    - Create a polygon and check if a point is inside it.
    - Create two polygons and check if they intersect.
    - Create a LineString with at least three points.
    - Create a list of points and a polygon. Check which points are inside the polygon.

    ## Summary

    - Shapely provides Python objects for points, lines, and polygons with properties like area, length, and centroid.
    - Spatial predicates (`within`, `intersects`, `touches`) test relationships between geometries.
    - Operations like `buffer`, `union`, and `difference` create new geometries from existing ones.
    - Shapely uses Cartesian coordinates, so ensure your geometries share the same projected coordinate system for accurate results.
    - These building blocks underpin GeoPandas, which extends them to work with collections of features.

    Next: [GeoPandas](https://cityseer.benchmarkurbanism.com/start/4-geopandas) shows how to manage and analyse many spatial features at once.
    """)
    return


if __name__ == "__main__":
    app.run()
