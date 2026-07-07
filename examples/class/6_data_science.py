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
    # Data Science

    This lesson applies two common data science techniques to urban data: dimensionality reduction with Principal Component Analysis (PCA) and prediction with a Random Forest model, both using `scikit-learn`. We first use `cityseer` to generate street network centrality and land use accessibility metrics, which then serve as the inputs for the analysis.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import osmnx as ox
    from cityseer.network import CityNetwork
    from cityseer.tools import io

    return CityNetwork, io, ox, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > The code below uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which wraps network preparation and metric computation behind a lean interface. For a full explanation of the `cityseer` workflow, see the [quickstart recipe](https://cityseer.benchmarkurbanism.com/examples/recipes/quickstart).

    First, we define the geographical area of interest (Nicosia, Cyprus) using longitude and latitude coordinates, and a buffer radius. Then, we use `cityseer` to:

    1. Create a buffered polygon area around the center point.
    2. Download and simplify the street network within this polygon from OpenStreetMap with `CityNetwork.from_osm`, which also builds the dual representation so that calculations are expressed per street segment (the [quickstart recipe](https://cityseer.benchmarkurbanism.com/examples/recipes/quickstart) explains primal vs. dual graphs).
    3. Calculate centrality measures (density, harmonic closeness, betweenness) at multiple distance thresholds using angular analysis; `density` is requested through the `closeness` expression dictionary since the default computes only harmonic closeness and betweenness.
    """)
    return


@app.cell
def _(CityNetwork, io):
    lng, lat = 33.36402, 35.17526
    buffer = 2000

    poly_wgs, _epsg_code = io.buffered_point_poly(lng, lat, buffer)
    cn = CityNetwork.from_osm(poly_wgs, to_crs_code=3035)
    cn.centrality_simplest(
        distances=[500, 1000, 2000, 5000],
        closeness={"density": "1", "harmonic": "1 / (1 + c / 90)"},
    )
    cn.nodes_gdf.head()
    return cn, poly_wgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we use `osmnx` to acquire data about restaurants within the previously defined polygonal area (`poly_wgs`).

    1. `ox.features_from_polygon` queries OpenStreetMap for features tagged with `"amenity": "restaurant"`.
    2. `.to_crs(epsg=3035)` reprojects the downloaded restaurant data to the ETRS89 / LAEA Europe projected coordinate system (EPSG:3035) to ensure consistency with the street network data and enable accurate spatial calculations.
    3. `gdf_rest[["amenity", "geometry"]]` filters out the `amenity` and `geometry` columns.
    4. `.reset_index(drop=True)` resets the DataFrame index for cleaner data handling.
    """)
    return


@app.cell
def _(ox, poly_wgs):
    gdf_rest = ox.features_from_polygon(poly_wgs, tags={"amenity": "restaurant"})
    gdf_rest = gdf_rest.to_crs(epsg=3035)
    gdf_rest = gdf_rest[["amenity", "geometry"]]
    gdf_rest = gdf_rest.reset_index(drop=True)
    gdf_rest.head()
    return (gdf_rest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then use `cityseer` once again to calculate accessibility to restaurants from each node in the street network.

    1. `cn.compute_accessibilities` takes the restaurant locations (`gdf_rest`) as input.
    2. `landuse_column_label="amenity"` specifies that the 'amenity' column in `gdf_rest` identifies the type of feature (restaurants in this case).
    3. `accessibility_keys=["restaurant"]` tells the method to calculate accessibility specifically for features labelled as 'restaurant'.
    4. `distances=[200, 400, 800]` specifies the distance thresholds (in meters) at which accessibility should be measured. For each node, this will count how many restaurants are reachable within 200m, 400m, and 800m along the network.

    The results are written to the network's nodes GeoDataFrame as new columns for these accessibility scores (e.g. `cc_restaurant_200` for the count of restaurants within 200m), and the method also returns the land use GeoDataFrame with assignment information. The snapshot taken with `cn.to_geopandas()` (here assigned to `nodes_gdf_acc`) joins all computed columns to the street geometries.
    """)
    return


@app.cell
def _(cn, gdf_rest):
    _cn, _gdf_rest = cn.compute_accessibilities(
        gdf_rest,
        landuse_column_label="amenity",
        accessibility_keys=["restaurant"],
        distances=[200, 400, 800],
    )
    nodes_gdf_acc = cn.to_geopandas()
    return (nodes_gdf_acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now arrive at the Data Science section, which performs Principal Component Analysis (PCA), a dimensionality reduction technique, on the street network centrality measures. The goal is to identify underlying patterns or latent dimensions within these (often correlated) centrality metrics.

    1. `StandardScaler()` is initialised to standardise the data. PCA is sensitive to the scale of variables, so standardisation (transforming data to have zero mean and unit variance) is a necessary preprocessing step.
    2. `X_scaled = scaler.fit_transform(...)` selects the twelve centrality columns (density, harmonic, and betweenness at four different distances) from `nodes_gdf_acc` and applies the standardisation.
    3. `pca = PCA(n_components=4)` initialises PCA to extract the top 4 principal components. These components are new, uncorrelated variables that capture the maximum possible variance from the original data.
    4. `X_pca = pca.fit_transform(X_scaled)` applies PCA to the scaled centrality data.
    5. The next four lines add these four principal components as new columns (`pca_1`, `pca_2`, `pca_3`, `pca_4`) to the `nodes_gdf_acc` GeoDataFrame.
    6. The subsequent code sets up a 3x1 plot to visualise the spatial distribution of the first three principal components. Each subplot shows the nodes coloured by their score on a principal component. The title of each subplot indicates the percentage of the original data's variance that is explained by that component. This helps in understanding how much information is retained by each component.
    """)
    return


@app.cell
def _(nodes_gdf_acc, plt):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        nodes_gdf_acc[
            [
                "cc_density_500_ang",
                "cc_density_1000_ang",
                "cc_density_2000_ang",
                "cc_density_5000_ang",
                "cc_harmonic_500_ang",
                "cc_harmonic_1000_ang",
                "cc_harmonic_2000_ang",
                "cc_harmonic_5000_ang",
                "cc_betweenness_500_ang",
                "cc_betweenness_1000_ang",
                "cc_betweenness_2000_ang",
                "cc_betweenness_5000_ang",
            ]
        ]
    )
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    nodes_gdf_acc["pca_1"] = X_pca[:, 0]
    nodes_gdf_acc["pca_2"] = X_pca[:, 1]
    nodes_gdf_acc["pca_3"] = X_pca[:, 2]
    nodes_gdf_acc["pca_4"] = X_pca[:, 3]
    _fig, _ax = plt.subplots(3, 1, figsize=(8, 24), dpi=150)
    nodes_gdf_acc.plot(
        column="pca_1", cmap="Reds", legend=True, legend_kwds={"label": "Component 1 score", "shrink": 0.6}, ax=_ax[0]
    )
    _ax[0].set_xlim(6433800, 6433800 + 2700)
    _ax[0].set_ylim(1669400, 1669400 + 2700)
    _ax[0].set_axis_off()
    _ax[0].set_title(f"Principal component 1 - explained variance: {pca.explained_variance_ratio_[0]:.0%}")
    nodes_gdf_acc.plot(
        column="pca_2", cmap="Reds", legend=True, legend_kwds={"label": "Component 2 score", "shrink": 0.6}, ax=_ax[1]
    )
    _ax[1].set_xlim(6433800, 6433800 + 2700)
    _ax[1].set_ylim(1669400, 1669400 + 2700)
    _ax[1].set_axis_off()
    _ax[1].set_title(f"Principal component 2 - explained variance: {pca.explained_variance_ratio_[1]:.0%}")
    nodes_gdf_acc.plot(
        column="pca_3", cmap="Reds", legend=True, legend_kwds={"label": "Component 3 score", "shrink": 0.6}, ax=_ax[2]
    )
    _ax[2].set_xlim(6433800, 6433800 + 2700)
    _ax[2].set_ylim(1669400, 1669400 + 2700)
    _ax[2].set_axis_off()
    _ax[2].set_title(f"Principal component 3 - explained variance: {pca.explained_variance_ratio_[2]:.0%}")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next cell uses `seaborn`, a statistical data visualisation library, to create a histogram and a joint plot.

    Histograms are useful for visualising the distribution of a single variable, while joint plots allow for the visualisation of the relationship between two variables along with their individual distributions.

    The histogram shows the distribution of restaurant accessibility within 800 meters (`cc_restaurant_800`). The x-axis represents the accessibility values, while the y-axis shows the frequency of these values. The `bins=50` argument specifies that the data should be divided into 50 bins for the histogram.

    The joint plot visualises the relationship between the first principal component (`pca_1`) and restaurant accessibility within 800 meters (`cc_restaurant_800`).
    """)
    return


@app.cell
def _(nodes_gdf_acc):
    import seaborn as sns

    _hist_ax = sns.histplot(data=nodes_gdf_acc, x="cc_restaurant_800", bins=50)
    _hist_ax.set_xlabel("Restaurant accessibility, 800 m")
    _joint_grid = sns.jointplot(data=nodes_gdf_acc, x="pca_1", y="cc_restaurant_800", kind="kde")
    _joint_grid.set_axis_labels("Principal component 1 score", "Restaurant accessibility, 800 m")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell demonstrates use of a Random Forest Regressor model to predict restaurant accessibility (`cc_restaurant_800`) based on the four principal components (`pca_1` to `pca_4`) derived from the network centrality measures. This is an example of supervised learning.

    1. `X` is defined as the DataFrame containing the predictor variables (the four PCA components).
    2. `y` is defined as the target variable (restaurant accessibility at 800m).
    3. `train_test_split(X, y, ...)` splits the data into training (80%) and testing (20%) sets. The model will be trained on the training set and evaluated on the unseen testing set. `random_state=42` ensures reproducibility of the split.
    4. `regressor = RandomForestRegressor(...)` initialises a Random Forest Regressor model. `n_estimators=100` means it will use 100 decision trees. `criterion="squared_error"` specifies the function to measure the quality of a split.
    5. `regressor.fit(X_train, y_train)` trains the model using the training data.
    6. `y_pred = regressor.predict(X_test)` makes predictions on the test set.
    7. `r2 = r2_score(y_test, y_pred)` calculates the R-squared score, a measure of how well the model's predictions fit the actual values in the test set. An R2 score closer to 1 indicates a better fit. Increasing the size of the training set generally improves the model's performance, as it has more data to learn from.
    8. The model then predicts accessibility for all nodes using `regressor.predict(X)` and stores these predictions in a new column `cc_restaurant_800_pred`.
    9. `nodes_gdf_acc["cc_restaurant_800_residuals"]` calculates the residuals (the difference between predicted and actual accessibility values).
    10. The final part of the cell sets up a 3x1 plot to visualise:
        - The actual restaurant accessibility.
        - The predicted restaurant accessibility (with the R2 score in the title).
        - The residuals of the regression, showing where the model over or under-predicts accessibility. A good model would have residuals randomly scattered around zero.
    """)
    return


@app.cell
def _(nodes_gdf_acc, plt):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    X = nodes_gdf_acc[["pca_1", "pca_2", "pca_3", "pca_4"]]
    y = nodes_gdf_acc["cc_restaurant_800"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, criterion="squared_error")
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R2 score: ", r2)
    nodes_gdf_acc["cc_restaurant_800_pred"] = regressor.predict(X)
    nodes_gdf_acc["cc_restaurant_800_residuals"] = (
        nodes_gdf_acc["cc_restaurant_800_pred"] - nodes_gdf_acc["cc_restaurant_800"]
    )
    _fig, _ax = plt.subplots(3, 1, figsize=(8, 24), dpi=150)
    nodes_gdf_acc.plot(
        column="cc_restaurant_800",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Restaurants within 800 m", "shrink": 0.6},
        ax=_ax[0],
    )
    _ax[0].set_xlim(6433800, 6433800 + 2700)
    _ax[0].set_ylim(1669400, 1669400 + 2700)
    _ax[0].set_axis_off()
    _ax[0].set_title("Restaurant accessibility, 800 m")
    nodes_gdf_acc.plot(
        column="cc_restaurant_800_pred",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Predicted restaurants within 800 m", "shrink": 0.6},
        ax=_ax[1],
    )
    _ax[1].set_xlim(6433800, 6433800 + 2700)
    _ax[1].set_ylim(1669400, 1669400 + 2700)
    _ax[1].set_axis_off()
    _ax[1].set_title(f"Predicted restaurant accessibility, 800 m - R2 score: {r2:.2f}")
    # coolwarm suits the residuals because they diverge around zero
    nodes_gdf_acc.plot(
        column="cc_restaurant_800_residuals",
        cmap="coolwarm",
        vmax=4,
        vmin=-4,
        legend=True,
        legend_kwds={"label": "Residual (predicted minus actual)", "shrink": 0.6},
        ax=_ax[2],
    )
    _ax[2].set_xlim(6433800, 6433800 + 2700)
    _ax[2].set_ylim(1669400, 1669400 + 2700)
    _ax[2].set_axis_off()
    _ax[2].set_title("Residuals of the Random Forest regression")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercises

    - Re-run the PCA with `n_components=2` instead of 4. How much total variance is explained by just two components? Plot the results.
    - Experiment with the Random Forest parameters: try changing `n_estimators` to 50 or 200, or adjust `test_size` to 0.3. How does the R² score change?
    - Replace the `RandomForestRegressor` with a different `scikit-learn` model (e.g., `GradientBoostingRegressor` or `LinearRegression`). Compare the R² scores — which model performs best?

    ## Summary

    - PCA reduces many correlated variables into a smaller set of uncorrelated principal components, each capturing a share of the original variance.
    - Standardising data before PCA is essential because PCA is sensitive to variable scales.
    - Supervised learning (e.g., Random Forest) trains on labelled data to predict outcomes for new observations.
    - R² scores measure how well predictions match actual values (1.0 = perfect, 0.0 = no better than the mean).
    - Residual maps reveal where a model over- or under-predicts, helping identify spatial patterns the model misses.

    These techniques combine well with `cityseer` network metrics; see the [Cityseer Recipes](https://cityseer.benchmarkurbanism.com/examples) section for practical analysis workflows.
    """)
    return


if __name__ == "__main__":
    app.run()
