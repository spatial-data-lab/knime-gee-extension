"""
GEE Tool Nodes for KNIME
This module contains utility nodes for Google Earth Engine data extraction and analysis.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

# Category for GEE Tool nodes
__category = knext.category(
    path="/community/gee",
    level_id="tool",
    name="Tool",
    description="Google Earth Engine Tool and Utility nodes",
    icon="icons/tool.png",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/tool/"


############################################
# Get Image Value by LatLon
############################################


@knext.node(
    name="Get Image Value by LatLon",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "latlonvalue.png",
    after="",
)
@knext.input_table(
    name="Input Table",
    description="Table containing ID, latitude, and longitude columns",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_table(
    name="Output Table",
    description="Table with ID and extracted image values for each band",
)
class GetImageValueByLatLon:
    """Extracts pixel values from a Google Earth Engine image at specified latitude/longitude coordinates.

    This node extracts pixel values from a Google Earth Engine image at specified latitude/longitude coordinates,
    creating a table with extracted values for each point. This node is useful for creating point-based datasets for statistical analysis,
    sampling remote sensing data for ground truth validation, and generating training data for machine learning models.
    The node uses efficient batch processing to handle large numbers of points quickly.

    **Input Requirements:**

    - Table must contain ID, latitude, and longitude columns

    - Coordinates should be in decimal degrees (WGS84)

    - Scale parameter controls sampling resolution (default: 30m)

    **Note:** Data transfer between local systems and Google Earth Engine cloud is subject to GEE's transmission limits.
    For large datasets (thousands of points), consider processing in smaller batches to avoid data limit errors.
    """

    id_column = knext.ColumnParameter(
        "ID Column",
        "Column containing unique identifiers for each point",
        port_index=0,
    )

    latitude_column = knext.ColumnParameter(
        "Latitude Column",
        "Column containing latitude values",
        port_index=0,
    )

    longitude_column = knext.ColumnParameter(
        "Longitude Column",
        "Column containing longitude values",
        port_index=0,
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters to use for sampling. Lower values provide higher resolution but may be slower.",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    def configure(self, configure_context, input_table_schema, input_binary_spec):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_table: knext.Table,
        image_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        # Get image directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        image = image_connection.gee_object

        # Convert input table to pandas DataFrame
        df = input_table.to_pandas()

        # Get image info to determine bands (optimized to only get band names)
        band_names = image.bandNames().getInfo()

        # LOGGER.warning(f"Processing {len(df)} points with {len(band_names)} bands")

        features = []
        for idx, row in df.iterrows():
            pt = ee.Geometry.Point(
                [float(row[self.longitude_column]), float(row[self.latitude_column])]
            )
            features.append(ee.Feature(pt, {"id": str(row[self.id_column])}))
        points_fc = ee.FeatureCollection(features)

        sampled = image.sampleRegions(
            collection=points_fc,
            properties=["id"],
            scale=self.scale,
            geometries=True,
        )

        sampled_info = sampled.getInfo()

        results = []
        for feature in sampled_info["features"]:
            point_id = feature["properties"]["id"]
            band_values = {}

            for band_name in band_names:
                band_values[band_name] = feature["properties"].get(band_name, None)

            results.append({"id": point_id, **band_values})

        # Create output DataFrame
        output_df = pd.DataFrame(results)

        # LOGGER.warning(f"Successfully extracted values for {len(output_df)} points")

        return knext.Table.from_pandas(output_df)


############################################
# Local GeoTable Reducer
############################################


@knext.node(
    name="Local GeoTable Reducer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "LocalTableReducer.png",
    after="",
)
@knext.input_table(
    name="Input GeoTable",
    description="Table containing geometry column for zonal statistics",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_table(
    name="Output Table",
    description="Table with zonal statistics for each geometry",
)
class LocalGeoTableReducer:
    """Performs zonal statistics on a Google Earth Engine image using local geometry data.

    This node performs zonal statistics on a Google Earth Engine image using local geometry data,
    calculating statistical summaries for each polygon, line, or point feature.
    This node is useful for calculating statistical summaries of raster values within vector boundaries,
    performing area-based analysis like average NDVI per administrative unit, generating summary
    statistics for environmental monitoring, and creating aggregated datasets for further analysis.
    Data transfer between local systems and Google Earth Engine cloud is subject to GEE's transmission limits.
    This node includes built-in batch processing functionality to handle large datasets efficiently.

    **Statistical Methods:**

    - **mean**: Average value within each geometry

    - **median**: Median value (robust to outliers)

    - **min/max**: Minimum/maximum values

    - **count**: Number of valid pixels

    - **sum**: Sum of all pixel values

    - **stdDev**: Standard deviation

    - **variance**: Statistical variance

    **Performance Features:**

    - **Batch Processing**: Handles large datasets by processing in chunks

    - **Configurable Scale**: Control sampling resolution for accuracy vs. speed

    - **Multiple Statistics**: Calculate several statistics simultaneously

    """

    geo_col = knext.ColumnParameter(
        "Geometry Column",
        "Column containing geometry data",
        column_filter=knut.is_geo,
        include_row_key=False,
        include_none_column=False,
        port_index=0,
    )

    reducer_methods = knext.StringParameter(
        "Reducer Methods",
        "Comma-separated list of reduction methods (e.g., 'mean,min,max')",
        default_value="mean",
    )

    image_scale = knext.IntParameter(
        "Image Scale (meters)",
        "The scale in meters for zonal statistics calculation",
        default_value=1000,
        min_value=1,
        max_value=10000,
    )

    batch_boolean = knext.BoolParameter(
        "Enable Batch Processing",
        "Enable batch processing for large datasets",
        default_value=False,
    )

    batch_size = knext.IntParameter(
        "Batch Size",
        "Number of features to process in each batch",
        default_value=100,
        min_value=1,
        max_value=10000,
    ).rule(knext.OneOf(batch_boolean, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_table_schema, input_binary_spec):
        self.geo_col = knut.column_exists_or_preset(
            configure_context, self.geo_col, input_table_schema, knut.is_geo
        )  # Show batch_size parameter only when batch_boolean is True
        if not self.batch_boolean:
            self.batch_size = 100  # Reset to default when batch is disabled
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_table: knext.Table,
        image_connection,
    ):
        import ee

        import geopandas as gp
        import geemap
        import pandas as pd

        # Get image directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        image = image_connection.gee_object

        # Map each reduction method to its corresponding ee.Reducer
        reducer_map = {
            "min": ee.Reducer.min(),
            "mean": ee.Reducer.mean(),
            "median": ee.Reducer.median(),
            "max": ee.Reducer.max(),
            "count": ee.Reducer.count(),
            "sum": ee.Reducer.sum(),
            "stdDev": ee.Reducer.stdDev(),
            "variance": ee.Reducer.variance(),
        }

        # Split the reducelist and create a combined reducer
        reduce_methods = [method.strip() for method in self.reducer_methods.split(",")]

        # Validate reducer methods
        valid_methods = []
        for method in reduce_methods:
            if method in reducer_map:
                valid_methods.append(method)

        if not valid_methods:
            raise ValueError("No valid reducer methods provided")

        # Create combined reducer
        reducers = reducer_map[valid_methods[0]]
        for method in valid_methods[1:]:
            reducers = reducers.combine(reducer2=reducer_map[method], sharedInputs=True)

        # Create GeoDataFrame
        shp = gp.GeoDataFrame(input_table.to_pandas(), geometry=self.geo_col)

        # Ensure CRS is WGS84 (EPSG:4326)
        if shp.crs is None:
            shp.set_crs(epsg=4326, inplace=True)
        else:
            shp.to_crs(4326, inplace=True)

        # Process based on batch setting
        if self.batch_boolean:

            def process_batch(batch):
                feature_collection = geemap.gdf_to_ee(batch)
                stats = image.reduceRegions(
                    collection=feature_collection,
                    reducer=reducers,
                    scale=self.image_scale,
                )
                return geemap.ee_to_gdf(stats)

            # Split into batches
            batches = [
                shp.iloc[i : i + self.batch_size]
                for i in range(0, len(shp), self.batch_size)
            ]

            # Process each batch
            result_dfs = []
            for i, batch in enumerate(batches):

                result_dfs.append(process_batch(batch))

            # Combine results
            result_df = pd.concat(result_dfs, ignore_index=True)
        else:

            # Convert to GEE Feature Collection
            feature_collection = geemap.gdf_to_ee(shp)

            # Perform zonal statistics
            stats = image.reduceRegions(
                collection=feature_collection, reducer=reducers, scale=self.image_scale
            )

            # Convert result to GeoDataFrame
            result_df = geemap.ee_to_gdf(stats)

        # Remove RowID column if present
        if "<RowID>" in result_df.columns:
            result_df = result_df.drop(columns=["<RowID>"])

        return knext.Table.from_pandas(result_df)
