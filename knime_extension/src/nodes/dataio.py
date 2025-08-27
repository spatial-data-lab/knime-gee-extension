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
@knext.output_binary(
    name="GEE Image",
    description="The output binary containing the GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
class GEEImageReader:
    """GEE Image Reader.
    Reads a single image from Google Earth Engine.
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

        # Get credentials and project ID from connection
        credentials = gee_connection.credentials
        project_id = gee_connection.spec.project_id

        # Initialize GEE with credentials and project

        ee.Initialize(credentials=credentials, project=project_id)

        # Load the image
        # LOGGER.warning(f"Loading image: {self.imagename}")
        image = ee.Image(self.imagename)

        # Get image info for logging
        # try:
        #     info = image.getInfo()
        #     LOGGER.warning(f"Successfully got image info: {info}")
        # except Exception as e:
        #     LOGGER.warning(f"Error getting image info: {e}")

        # LOGGER.warning(f"Image object type: {type(image)}")
        # LOGGER.warning(f"Image object: {image}")

        # Export with credentials
        output_binary = knut.gee_export_init(image, credentials, project_id)
        # LOGGER.warning(f"Image binary size: {len(output_binary)} bytes")

        return output_binary


@knext.node(
    name="GEE Collection Image Reader",
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
@knext.output_binary(
    name="GEE Image",
    description="The output binary containing the aggregated GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
class GEEImageCollectionReader:
    """GEE Image Collection Reader.
    Reads and aggregates images from a Google Earth Engine collection.
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

        # Get credentials and project ID from connection
        credentials = gee_connection.credentials
        project_id = gee_connection.spec.project_id

        # Initialize GEE
        ee.Initialize(credentials=credentials, project=project_id)

        # Parse dates
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=self.date_span)

        LOGGER.warning(f"Filtering collection from {start_date} to {end_date}")

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

        # Export with credentials
        output_binary = knut.gee_export_init(image, credentials, project_id)
        return output_binary


@knext.node(
    name="Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="GEE Image",
    description="The input binary containing the GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="GEE Image",
    description="The output binary containing the filtered GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
class BandSelector:
    """Band Selector.
    Selects specific bands from a GEE Image.
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
        input_binary,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Use the standardized import function
        credentials, project_id, image = knut.gee_import_init(input_binary, LOGGER)

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

        # Export with credentials
        output = knut.gee_export_init(image, credentials, project_id)
        return output


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
@knext.input_binary(
    name="GEE Image",
    description="The input binary containing the GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
@knext.output_table(
    name="Output Table",
    description="Table with ID and extracted image values for each band",
)
class GetImageValueByLatLon:
    """Get Image Value by LatLon.
    Extracts pixel values from a GEE image at specified latitude/longitude coordinates.
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
        input_binary,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        # Use the standardized import function
        credentials, project_id, image = knut.gee_import_init(input_binary, LOGGER)

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


@knext.node(
    name="Feature Collection to Table",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "io.png",
    after="",
)
@knext.input_binary(
    name="GEE Feature Collection",
    description="The input binary containing the GEE Feature Collection with embedded credentials.",
    id="geemap.gee.FeatureCollection",
)
@knext.output_table(
    name="Output Table",
    description="Table converted from GEE Feature Collection",
)
class FeatureCollectionToTable:
    """Feature Collection to Table.
    Converts a GEE Feature Collection to a KNIME table.
    """

    file_format = knext.StringParameter(
        "Output Format",
        "Format for the output table",
        default_value="DataFrame",
        enum=["DataFrame", "GeoDataFrame"],
    )

    def configure(self, configure_context, input_binary_spec):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_binary,
    ):
        import ee
        import logging
        import pandas as pd
        import geemap

        # LOGGER = logging.getLogger(__name__)

        # Use the standardized import function for feature collection
        credentials, project_id, feature_collection = knut.gee_import_init(input_binary)

        # LOGGER.warning(f"Converting Feature Collection to {self.file_format}")

        # Convert based on format
        if self.file_format == "DataFrame":
            df = geemap.ee_to_df(feature_collection)
        else:  # GeoDataFrame
            df = geemap.ee_to_gdf(feature_collection)

        # Remove RowID column if present
        if "<RowID>" in df.columns:
            df = df.drop(columns=["<RowID>"])
            # LOGGER.warning("Removed <RowID> column from output")

        # LOGGER.warning(
        #     f"Successfully converted Feature Collection to table with {len(df)} rows"
        # )

        return knext.Table.from_pandas(df)


@knext.node(
    name="GeoTable to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "io.png",
    after="",
)
@knext.input_table(
    name="Input GeoTable",
    description="Table containing geometry column for conversion to Feature Collection",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_binary(
    name="GEE Feature Collection",
    description="The output binary containing the GEE Feature Collection with embedded credentials.",
    id="geemap.gee.FeatureCollection",
)
class GeoTableToFeatureCollection:
    """GeoTable to Feature Collection.
    Converts a KNIME table with geometry to a GEE Feature Collection.
    """

    geo_col = knext.ColumnParameter(
        "Geometry Column",
        "Column containing geometry data",
        port_index=0,
    )

    def configure(self, configure_context, input_table_schema, input_schema):
        self.geo_col = knut.column_exists_or_preset(
            configure_context, self.geo_col, input_table_schema, knut.is_geo
        )
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_table: knext.Table,
        gee_connection,
    ):
        import ee
        import logging
        import geopandas as gp
        import geemap

        LOGGER = logging.getLogger(__name__)

        # Get credentials and project ID from connection
        credentials = gee_connection.credentials
        project_id = gee_connection.spec.project_id

        # Initialize GEE with credentials and project
        ee.Initialize(credentials=credentials, project=project_id)

        # Convert input table to pandas DataFrame
        df = input_table.to_pandas()

        # LOGGER.warning(f"Converting table with {len(df)} rows to Feature Collection")

        # Create GeoDataFrame
        shp = gp.GeoDataFrame(df, geometry=self.geo_col)

        # Ensure CRS is WGS84 (EPSG:4326)
        if shp.crs is None:
            shp.set_crs(epsg=4326, inplace=True)
        else:
            shp.to_crs(4326, inplace=True)

        # LOGGER.warning(f"GeoDataFrame CRS: {shp.crs}")

        # Convert to GEE Feature Collection
        feature_collection = geemap.gdf_to_ee(shp)

        # Export with credentials
        output_binary = knut.gee_export_init(
            feature_collection, credentials, project_id
        )

        return output_binary


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
@knext.input_binary(
    name="GEE Image",
    description="The input binary containing the GEE Image with embedded credentials.",
    id="geemap.gee.Image",
)
@knext.output_table(
    name="Output Table",
    description="Table with zonal statistics for each geometry",
)
class LocalGeoTableReducer:
    """Local GeoTable Reducer.
    Performs zonal statistics on GEE image using local geometry table.
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
    )

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
        input_binary,
    ):
        import ee

        import geopandas as gp
        import geemap
        import pandas as pd

        # Use the standardized import function for image
        credentials, project_id, image = knut.gee_import_init(input_binary)

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
