"""
GEE Data I/O Nodes for KNIME
This module contains nodes for reading, filtering, and extracting data from Google Earth Engine.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

# Category for GEE Data I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="dataio",
    name="Data I/O",
    description="Google Earth Engine Data Input/Output nodes",
    icon="icons/dataset.png",
)

# Node icon path
__NODE_ICON_PATH = "icons/"


############################################
# GEE Image Reader
############################################


@knext.node(
    name="GEE Image Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "dataset.png",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
class GEEImageReader:
    """Loads a single image from Google Earth Engine using the specified image ID.

    This node allows you to access individual satellite images, elevation data, or other geospatial datasets from GEE's
    extensive catalog for further analysis in KNIME workflows.

    **Common Image Examples:**

    - Elevation: 'USGS/SRTMGL1_003' (30m resolution)

    - ESA Elevation: 'ESA/WorldCover/v100' (10m resolution)

    - WorldPop Population: 'CIESIN/GPWv411/GPW_Population_Density' (30 arc-second)

    - Global Forest: 'UMD/hansen/global_forest_change_2021_v1_9' (30m resolution)

    - Global Settlement: 'WSF/WSF_v1' (10m resolution)
    """

    imagename = knext.StringParameter(
        "Image Name",
        "The name/ID of the GEE image to load (e.g., 'USGS/SRTMGL1_003')",
        default_value="USGS/SRTMGL1_003",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee

        image = ee.Image(self.imagename)

        return knut.export_gee_connection(image, gee_connection)


############################################
# GEE Image Collection Reader
############################################


@knext.node(
    name="GEE Image Collection Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "dataset.png",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded aggregated image object.",
    port_type=google_earth_engine_port_type,
)
class GEEImageCollectionReader:
    """Loads and aggregates multiple images from a Google Earth Engine image collection based on date range and aggregation method.

    This node is useful for accessing time-series satellite data, reducing cloud cover by creating composite images,
    and processing large collections of satellite imagery efficiently.

    **Aggregation Methods:**

    - **first**: Returns the first image in the collection (useful for cloud-free composites)

    - **mean**: Calculates pixel-wise mean (good for reducing noise)

    - **median**: Calculates pixel-wise median (robust to outliers)

    - **min/max**: Finds minimum/maximum values (useful for NDVI analysis)

    - **sum**: Adds pixel values (useful for cumulative indices)

    - **mode**: Finds most frequent values (useful for classification)

    **Common Collections:**

    - Sentinel-2: 'COPERNICUS/S2_SR' (10m resolution, optical)

    - Landsat 8: 'LANDSAT/LC08/C02/T1_L2' (30m resolution, optical)

    - MODIS: 'MODIS/006/MOD13Q1' (250m resolution, vegetation indices)

    - Landsat 7: 'LANDSAT/LE07/C02/T1_L2' (30m resolution, optical)

    - Sentinel-1: 'COPERNICUS/S1_GRD' (10m resolution, radar)
    """

    collection_id = knext.StringParameter(
        "Collection ID",
        "The ID of the GEE image collection (e.g., 'COPERNICUS/S2_SR')",
        default_value="COPERNICUS/S2_SR",
    )

    start_date = knext.StringParameter(
        "Start Date",
        "Start date in YYYY-MM-DD format",
        default_value="2024-01-01",
    )

    date_span = knext.IntParameter(
        "Date Span (days)",
        "Number of days to include after start date",
        default_value=30,
        min_value=1,
        max_value=365,
    )

    aggregation_method = knext.StringParameter(
        "Aggregation Method",
        "Method to aggregate multiple images",
        default_value="first",
        enum=["first", "mean", "median", "min", "max", "sum", "mode"],
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import logging
        from datetime import datetime, timedelta

        LOGGER = logging.getLogger(__name__)

        # GEE is already initialized in the same Python process from the connection

        # Parse dates
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=self.date_span)

        # LOGGER.warning(f"Filtering collection from {start_date} to {end_date}")

        # Filter image collection by date
        image_collection = ee.ImageCollection(self.collection_id).filterDate(
            start_date, end_date
        )

        # Define aggregation methods
        aggregation_methods = {
            "first": image_collection.first,
            "mean": image_collection.mean,
            "median": image_collection.median,
            "min": image_collection.min,
            "max": image_collection.max,
            "sum": image_collection.sum,
            "mode": image_collection.mode,
        }

        # Apply aggregation
        try:
            image = aggregation_methods[self.aggregation_method]()
            # LOGGER.warning(
            #     f"Successfully aggregated using {self.aggregation_method} method"
            # )
        except Exception as e:
            LOGGER.warning(
                f"Aggregation method '{self.aggregation_method}' failed, falling back to 'first'. Error: {e}"
            )
            image = image_collection.first()

        return knut.export_gee_connection(image, gee_connection)


############################################
# Image Band Selector
############################################
@knext.node(
    name="Image Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with filtered image object.",
    port_type=google_earth_engine_port_type,
)
class BandSelector:
    """Filters and selects specific bands from a Google Earth Engine image.

    This node allows you to filter and select specific bands from a Google Earth Engine image, allowing you to focus on relevant spectral information and reduce data size.
    This node is useful for preparing images for specific applications like vegetation analysis, water detection, or optimizing processing speed by selecting only necessary bands.

    **Common Band Combinations:**

    - **RGB**: 'B4,B3,B2' (Sentinel-2) or 'B4,B3,B2' (Landsat 8)

    - **False Color**: 'B8,B4,B3' (Sentinel-2) - good for vegetation

    - **SWIR**: 'B12,B8,B4' (Sentinel-2) - good for moisture detection

    - **NDVI Bands**: 'B8,B4' (Sentinel-2) - for vegetation index calculation


    **Note:** If no specified bands are found in the image, the original image is returned unchanged.
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of band names to select (e.g., 'B1,B2,B3'). Leave empty to keep all bands.",
        default_value="",
    )

    def configure(self, configure_context, input_binary_spec):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        image = image_connection.gee_object

        # Get original image info for logging
        original_info = image.getInfo()
        original_bands = [band["id"] for band in original_info.get("bands", [])]
        # LOGGER.warning(f"Original image bands: {original_bands}")

        band_list = [band.strip() for band in self.bands.split(",")]
        # LOGGER.warning(f"Selecting bands: {band_list}")

        # Filter to only include bands that exist in the image
        available_bands = [band for band in band_list if band in original_bands]
        if available_bands:
            image = image.select(available_bands)
            # LOGGER.warning(f"Successfully selected bands: {available_bands}")
        else:
            LOGGER.warning(
                f"No specified bands found in image. Available bands: {original_bands}"
            )
            # If no bands match, return original image

        return knut.export_gee_connection(image, image_connection)


############################################
# Get Image Value by LatLon
############################################
@knext.node(
    name="Get Image Value by LatLon",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "io.png",
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

        # Get image info to determine bands
        image_info = image.getInfo()
        bands = image_info.get("bands", [])
        band_names = [band["id"] for band in bands]

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
    icon_path=__NODE_ICON_PATH + "io.png",
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
