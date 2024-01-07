---
layout: ../../layouts/PageLayout.astro
---
<section class="module">

# visibility


 Visibility and viewshed analysis.


<div class="function">

## visibility_graph


<div class="content">
<span class="name">visibility_graph</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">bldgs_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">bounds</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">out_path</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">from_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">resolution</span>
    <span class="pc">:</span>
    <span class="pa"> int = 1</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Run a visibility graph analysis. This will return three TIFF image files, respectively showing a density, farness, and harmonic closeness based measure.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">bldgs_gdf</div>
    <div class="type">gpd.GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame containing building polygons.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">bounds</div>
    <div class="type">tuple[float, float, float, float]</div>
  </div>
  <div class="desc">

 A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">out_path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An output path to which the generated TIFF images will be written. The pathname will be appended to correspond to the density, farness, and harmonic closeness measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">from_crs_code</div>
    <div class="type">int | str</div>
  </div>
  <div class="desc">

 The EPSG coordinate reference code corresponding to the input data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int | str | None = None</div>
  </div>
  <div class="desc">

 An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">view_distance</div>
    <div class="type">int = 100</div>
  </div>
  <div class="desc">

 The view distance within which to run the visibility analysis. 100m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">resolution</div>
    <div class="type">int = 1</div>
  </div>
  <div class="desc">

 The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster analysis.</div>
</div>


</div>


<div class="function">

## visibility_graph_from_osm


<div class="content">
<span class="name">visibility_graph_from_osm</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">bounds_wgs</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">out_path</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">resolution</span>
    <span class="pc">:</span>
    <span class="pa"> int = 1</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Retrieves OSM buildings for the specified WGS bounds and runs a visibility analysis. This will return three TIFF image files, respectively showing a density, farness, and harmonic closeness based measure.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">bounds_wgs</div>
    <div class="type">tuple[float, float, float, float]</div>
  </div>
  <div class="desc">

 A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">out_path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An output path to which the generated TIFF images will be written. The pathname will be appended to correspond to the density, farness, and harmonic closeness measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int | str | None = None</div>
  </div>
  <div class="desc">

 An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">view_distance</div>
    <div class="type">int = 100</div>
  </div>
  <div class="desc">

 The view distance within which to run the visibility analysis. 100m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">resolution</div>
    <div class="type">int = 1</div>
  </div>
  <div class="desc">

 The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster analysis.</div>
</div>


</div>


<div class="function">

## viewshed


<div class="content">
<span class="name">viewshed</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">bldgs_gdf</span>
    <span class="pc">:</span>
    <span class="pa"> geopandas.geodataframe.GeoDataFrame</span>
  </div>
  <div class="param">
    <span class="pn">bounds</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">origin_x</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">origin_y</span>
    <span class="pc">:</span>
    <span class="pa"> float</span>
  </div>
  <div class="param">
    <span class="pn">out_path</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">from_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">resolution</span>
    <span class="pc">:</span>
    <span class="pa"> int = 1</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Run a viewshed analysis from a specified point. Writes an output image to the specified output path.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">bldgs_gdf</div>
    <div class="type">gpd.GeoDataFrame</div>
  </div>
  <div class="desc">

 A GeoDataFrame containing building polygons.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">bounds</div>
    <div class="type">tuple[float, float, float, float]</div>
  </div>
  <div class="desc">

 A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_x</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 An easting or longitude for the origin of the viewshed in the `from_crs_code` coordinate reference system.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_y</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A northing or latitude for the origin of the viewshed in the `from_crs_code` coordinate reference system.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">out_path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An output path to which the generated TIFF images will be written. The pathname will be appended to correspond to the density, farness, and harmonic closeness measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">from_crs_code</div>
    <div class="type">int | str</div>
  </div>
  <div class="desc">

 The EPSG coordinate reference code corresponding to the input data.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int | str | None = None</div>
  </div>
  <div class="desc">

 An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">view_distance</div>
    <div class="type">int = 100</div>
  </div>
  <div class="desc">

 The view distance within which to run the visibility analysis. 100m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">resolution</div>
    <div class="type">int = 1</div>
  </div>
  <div class="desc">

 The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster analysis.</div>
</div>


</div>


<div class="function">

## viewshed_from_osm


<div class="content">
<span class="name">viewshed_from_osm</span><div class="signature multiline">
  <span class="pt">(</span>
  <div class="param">
    <span class="pn">bounds_wgs</span>
    <span class="pc">:</span>
    <span class="pa"> tuple[float, float, float, float]</span>
  </div>
  <div class="param">
    <span class="pn">origin_lng</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">origin_lat</span>
    <span class="pc">:</span>
    <span class="pa"> int</span>
  </div>
  <div class="param">
    <span class="pn">out_path</span>
    <span class="pc">:</span>
    <span class="pa"> str</span>
  </div>
  <div class="param">
    <span class="pn">to_crs_code</span>
    <span class="pc">:</span>
    <span class="pa"> int | str | None = None</span>
  </div>
  <div class="param">
    <span class="pn">view_distance</span>
    <span class="pc">:</span>
    <span class="pa"> int = 100</span>
  </div>
  <div class="param">
    <span class="pn">resolution</span>
    <span class="pc">:</span>
    <span class="pa"> int = 1</span>
  </div>
  <span class="pt">)</span>
</div>
</div>


 Run a viewshed analysis from a specified point using OSM data. Writes an output image to the specified output path.
### Parameters
<div class="param-set">
  <div class="def">
    <div class="name">bounds_wgs</div>
    <div class="type">tuple[float, float, float, float]</div>
  </div>
  <div class="desc">

 A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_lng</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A longitude for the origin of the viewshed in WGS84 coordinates.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">origin_lat</div>
    <div class="type">float</div>
  </div>
  <div class="desc">

 A latitude for the origin of the viewshed in WGS84 coordinates.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">out_path</div>
    <div class="type">str</div>
  </div>
  <div class="desc">

 An output path to which the generated TIFF images will be written. The pathname will be appended to correspond to the density, farness, and harmonic closeness measures.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">to_crs_code</div>
    <div class="type">int | str | None = None</div>
  </div>
  <div class="desc">

 An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">view_distance</div>
    <div class="type">int = 100</div>
  </div>
  <div class="desc">

 The view distance within which to run the visibility analysis. 100m by default.</div>
</div>

<div class="param-set">
  <div class="def">
    <div class="name">resolution</div>
    <div class="type">int = 1</div>
  </div>
  <div class="desc">

 The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster analysis.</div>
</div>


</div>



</section>
