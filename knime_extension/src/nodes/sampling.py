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
    level_id="sampling",
    name="Sampling",
    description="Google Earth Engine Sampling nodes",
    icon="icons/Sampling.png",
    after="visualize",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/sampling/"


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

    **Batch Processing:**

    For large point datasets, batch processing is automatically enabled to handle
    GEE's data transfer limits (~16MB). The node processes points in batches and
    combines the results automatically.

    **Note:** Data transfer between local systems and Google Earth Engine cloud is subject to GEE's transmission limits.
    Batch processing helps avoid these limits for large datasets.
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

    batch_size = knext.IntParameter(
        "Batch Size",
        "Number of points to process in each batch (smaller batches = safer but slower). Batch processing is automatically enabled for large datasets.",
        default_value=500,
        min_value=50,
        max_value=5000,
        is_advanced=True,
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

        LOGGER.warning(f"Processing {len(df)} points with {len(band_names)} bands")

        # Use batch processing if dataset is large (auto-detect)
        use_batch = len(df) > 1000

        try:
            if use_batch:
                LOGGER.warning(
                    f"Using batch processing with batch size {self.batch_size}"
                )
                output_df = self._extract_values_with_batch(
                    df,
                    image,
                    band_names,
                    self.scale,
                    self.batch_size,
                    LOGGER,
                    exec_context,
                )
            else:
                # Direct processing for small datasets
                output_df = self._extract_values_direct(
                    df, image, band_names, self.scale
                )

            LOGGER.warning(f"Successfully extracted values for {len(output_df)} points")

            return knext.Table.from_pandas(output_df)

        except Exception as e:
            # If direct processing fails, try batch processing
            if not use_batch:
                LOGGER.warning(
                    f"Direct processing failed ({e}), retrying with batch processing"
                )
                try:
                    output_df = self._extract_values_with_batch(
                        df,
                        image,
                        band_names,
                        self.scale,
                        self.batch_size,
                        LOGGER,
                        exec_context,
                    )
                    LOGGER.warning(
                        f"Successfully extracted values for {len(output_df)} points (using batch processing)"
                    )
                    return knext.Table.from_pandas(output_df)
                except Exception as batch_error:
                    LOGGER.error(f"Batch processing also failed: {batch_error}")
                    raise
            else:
                LOGGER.error(f"Extract image values failed: {e}")
                raise

    def _extract_values_direct(self, df, image, band_names, scale):
        """Extract values directly without batch processing"""
        import ee
        import pandas as pd

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
            scale=scale,
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

        return pd.DataFrame(results)

    def _extract_values_with_batch(
        self, df, image, band_names, scale, batch_size, logger, exec_context=None
    ):
        """Extract values using batch processing"""
        import ee
        import pandas as pd

        total_size = len(df)
        num_batches = (total_size + batch_size - 1) // batch_size
        all_results = []

        for i in range(num_batches):
            # Update progress
            if exec_context is not None:
                progress = 0.1 + (i / num_batches) * 0.7
                exec_context.set_progress(
                    progress, f"Processing batch {i + 1}/{num_batches}"
                )

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_size)

            # Get batch of points
            batch_df = df.iloc[start_idx:end_idx]

            try:
                # Create Feature Collection for this batch
                features = []
                for idx, row in batch_df.iterrows():
                    pt = ee.Geometry.Point(
                        [
                            float(row[self.longitude_column]),
                            float(row[self.latitude_column]),
                        ]
                    )
                    features.append(ee.Feature(pt, {"id": str(row[self.id_column])}))
                batch_fc = ee.FeatureCollection(features)

                # Sample regions
                sampled = image.sampleRegions(
                    collection=batch_fc,
                    properties=["id"],
                    scale=scale,
                    geometries=True,
                )

                # Get results
                sampled_info = sampled.getInfo()

                batch_results = []
                for feature in sampled_info["features"]:
                    point_id = feature["properties"]["id"]
                    band_values = {}

                    for band_name in band_names:
                        band_values[band_name] = feature["properties"].get(
                            band_name, None
                        )

                    batch_results.append({"id": point_id, **band_values})

                all_results.extend(batch_results)
                logger.warning(
                    f"Processed batch {i + 1}/{num_batches} ({len(batch_df)} points)"
                )

            except Exception as e:
                logger.warning(
                    f"Batch {i + 1} failed: {e}, trying smaller batch size..."
                )
                # Try with smaller batches for this range
                smaller_batch = max(len(batch_df) // 2, 50)
                sub_results = []

                for j in range(start_idx, end_idx, smaller_batch):
                    sub_end_idx = min(j + smaller_batch, end_idx)
                    sub_batch_df = df.iloc[j:sub_end_idx]

                    try:
                        sub_features = []
                        for idx, row in sub_batch_df.iterrows():
                            pt = ee.Geometry.Point(
                                [
                                    float(row[self.longitude_column]),
                                    float(row[self.latitude_column]),
                                ]
                            )
                            sub_features.append(
                                ee.Feature(pt, {"id": str(row[self.id_column])})
                            )
                        sub_batch_fc = ee.FeatureCollection(sub_features)

                        sub_sampled = image.sampleRegions(
                            collection=sub_batch_fc,
                            properties=["id"],
                            scale=scale,
                            geometries=True,
                        )

                        sub_sampled_info = sub_sampled.getInfo()

                        for feature in sub_sampled_info["features"]:
                            point_id = feature["properties"]["id"]
                            band_values = {}

                            for band_name in band_names:
                                band_values[band_name] = feature["properties"].get(
                                    band_name, None
                                )

                            sub_results.append({"id": point_id, **band_values})

                    except Exception as sub_error:
                        logger.error(
                            f"Sub-batch starting at {j} failed: {sub_error}, skipping..."
                        )
                        # Skip this sub-batch and continue
                        continue

                all_results.extend(sub_results)

        # Set final progress
        if exec_context is not None:
            exec_context.set_progress(0.8, "Combining results...")

        return pd.DataFrame(all_results)


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


############################################
# Reduce Regions (Server-side Zonal Statistics)
############################################


@knext.node(
    name="Feature Collection Reducer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "RegionReducer.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with regions for statistics.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with statistics added as properties.",
    port_type=google_earth_engine_port_type,
)
class ReduceRegions:
    """Performs server-side zonal statistics on GEE images using GEE Feature Collections.

    This node calculates statistical summaries of image values within each feature
    of a Feature Collection, all computed on the GEE server to avoid data transfer
    limitations. The results are added as properties to the Feature Collection.
    Use the **Feature Collection to Table** node to convert the results to a table format.
    This design is essential for large-scale analysis and avoids downloading
    large datasets to local systems.

    **Statistical Methods:**

    - **mean**: Average value within each region
    - **median**: Median value (robust to outliers)
    - **min/max**: Minimum/maximum values
    - **count**: Number of valid pixels
    - **sum**: Sum of all pixel values
    - **stdDev**: Standard deviation
    - **variance**: Statistical variance
    - **percentile**: Custom percentile values

    **Performance Features:**

    - **Server-side Processing**: All computation on GEE servers
    - **Best Effort**: Automatically handles large regions
    - **Tile Scale**: Configurable for performance optimization
    - **Multiple Statistics**: Calculate several statistics simultaneously

    **Common Use Cases:**

    - Calculate average NDVI per administrative unit
    - Analyze land cover statistics by watershed
    - Compute climate statistics by region
    - Generate summary statistics for large areas
    - Avoid data transfer limits for big datasets

    **Best Practices:**

    - Use appropriate scale for your analysis needs
    - Enable bestEffort for large or complex regions
    - Adjust tileScale for performance optimization
    - Combine multiple statistics in one operation
    """

    reducer_methods = knext.StringParameter(
        "Reducer Methods",
        "Comma-separated list of reduction methods (e.g., 'mean,min,max,stdDev')",
        default_value="mean",
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters for zonal statistics calculation",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    tile_scale = knext.DoubleParameter(
        "Tile Scale",
        "Tile scale for performance optimization (1.0 = default, higher = faster but less precise)",
        default_value=1.0,
        min_value=0.1,
        max_value=16.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        fc_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and feature collection from connections
            image = image_connection.gee_object
            feature_collection = fc_connection.gee_object

            # Parse reducer methods
            reduce_methods = [
                method.strip() for method in self.reducer_methods.split(",")
            ]

            # Create combined reducer
            reducers = self._create_combined_reducer(reduce_methods)

            # Perform reduceRegions
            stats = image.reduceRegions(
                collection=feature_collection,
                reducer=reducers,
                scale=self.scale,
                tileScale=self.tile_scale,
            )

            LOGGER.warning(
                f"Successfully calculated zonal statistics using methods: {reduce_methods}"
            )

            # Return Feature Collection instead of table
            # User can use Feature Collection to Table node to convert if needed
            return knut.export_gee_connection(stats, fc_connection)

        except Exception as e:
            LOGGER.error(f"Reduce regions failed: {e}")
            raise

    def _create_combined_reducer(self, methods):
        """Create a combined reducer from multiple methods"""
        import ee

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

        # Validate and create combined reducer
        valid_methods = []
        for method in methods:
            if method in reducer_map:
                valid_methods.append(method)

        if not valid_methods:
            raise ValueError("No valid reducer methods provided")

        # Create combined reducer
        reducers = reducer_map[valid_methods[0]]
        for method in valid_methods[1:]:
            reducers = reducers.combine(reducer2=reducer_map[method], sharedInputs=True)

        return reducers


############################################
# Count by Class
############################################


@knext.node(
    name="Count by Class",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "CountClass.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with classified image (single band with class codes).",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with regions for counting pixels by class.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with pixel counts for each class added as properties.",
    port_type=google_earth_engine_port_type,
)
class CountByClass:
    """Counts pixels by classification class within each feature of a Feature Collection.

    This node calculates the number of pixels for each specified classification class
    within each feature of a Feature Collection. This is useful for analyzing land cover
    composition, calculating class proportions, and generating statistics for classification results.

    **How it Works:**

    - Automatically detects all unique class codes from the image (default) or uses manually specified codes
    - For each class code, creates a binary mask (1 = class, 0 = not class)
    - Uses reduceRegions with sum reducer to count pixels of each class per feature
    - Adds pixel counts as properties to each feature (e.g., 'class_1', 'class_2', etc.)

    **Class Detection:**

    - **Auto-detect (default)**: Automatically finds all unique class codes in the image
      by sampling pixels within the Feature Collection region. This is convenient
      and ensures all classes are counted.
    - **Manual**: Specify class codes manually as a comma-separated list (e.g., '1,2,3,4,5').
      Useful when you only want to count specific classes or when auto-detection fails.

    **Common Use Cases:**

    - Calculate land cover class proportions within administrative boundaries
    - Analyze classification results by region
    - Generate class distribution statistics for training sample validation
    - Calculate class coverage percentages within study areas

    **Input Requirements:**

    - Image must be a classified image (single band with integer class codes)
    - Feature Collection should contain polygon features representing regions
    - Class codes should match the values in your classified image

    **Output Format:**

    Each feature in the output Feature Collection will have additional properties:
    - 'class_X': Pixel count for class code X
    - Original feature properties are preserved
    """

    auto_detect_classes = knext.BoolParameter(
        "Auto-detect All Classes",
        "Automatically detect all unique class codes from the image",
        default_value=True,
    )

    category_codes = knext.StringParameter(
        "Class Codes",
        "Comma-separated list of class codes to count (e.g., '1,2,3,4,5'). Only used when auto-detect is disabled.",
        default_value="1,2,3",
    ).rule(knext.OneOf(auto_detect_classes, [False]), knext.Effect.SHOW)

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters for pixel counting",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    tile_scale = knext.DoubleParameter(
        "Tile Scale",
        "Tile scale for performance optimization (1.0 = default, higher = faster but less precise)",
        default_value=1.0,
        min_value=0.1,
        max_value=16.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and feature collection from connections
            image = image_connection.gee_object
            feature_collection = fc_connection.gee_object

            # Get category codes
            if self.auto_detect_classes:
                # Auto-detect all unique class codes from the image
                LOGGER.warning("Auto-detecting class codes from image...")

                # Get the ROI (union of all features)
                roi = feature_collection.geometry()

                # Get the first band (assuming single-band classified image)
                band_names = image.bandNames().getInfo()
                if not band_names:
                    raise ValueError("Image has no bands")

                band_name = band_names[0]
                image_band = image.select(band_name)

                # Method 1: Use sampling to get unique values
                # This is more reliable than histogram for classified images
                try:
                    sample = image_band.sample(
                        region=roi,
                        scale=self.scale,
                        numPixels=50000,  # Sample more pixels for better coverage
                        seed=42,
                    )

                    # Get unique values from sampled pixels
                    values = sample.aggregate_array(band_name).getInfo()

                    # Filter out None values and convert to integers
                    category_codes = sorted(
                        list(set([int(float(v)) for v in values if v is not None]))
                    )

                    if category_codes:
                        LOGGER.warning(
                            f"Auto-detected {len(category_codes)} classes using sampling: {category_codes}"
                        )
                    else:
                        raise ValueError("No valid class codes found in sample")

                except Exception as sampling_error:
                    LOGGER.warning(
                        f"Sampling method failed: {sampling_error}, trying histogram method..."
                    )

                    # Method 2: Fallback to histogram
                    try:
                        histogram = image_band.reduceRegion(
                            reducer=ee.Reducer.frequencyHistogram(),
                            geometry=roi,
                            scale=self.scale,
                            maxPixels=1e9,
                            bestEffort=True,
                        )

                        hist_dict = histogram.getInfo()

                        if hist_dict and band_name in hist_dict:
                            band_hist = hist_dict[band_name]
                            # Extract all keys (class codes) from the histogram
                            category_codes = sorted(
                                [int(float(k)) for k in band_hist.keys()]
                            )

                            if category_codes:
                                LOGGER.warning(
                                    f"Auto-detected {len(category_codes)} classes using histogram: {category_codes}"
                                )
                            else:
                                raise ValueError(
                                    "No valid class codes found in histogram"
                                )
                        else:
                            raise ValueError("Histogram returned empty or invalid data")

                    except Exception as hist_error:
                        raise ValueError(
                            f"Could not auto-detect class codes from image. "
                            f"Sampling error: {sampling_error}. "
                            f"Histogram error: {hist_error}. "
                            f"Please disable auto-detect and specify class codes manually."
                        )
            else:
                # Parse manually specified category codes
                category_codes_str = self.category_codes.strip()
                if not category_codes_str:
                    raise ValueError(
                        "Class codes cannot be empty when auto-detect is disabled"
                    )

                category_codes = [
                    int(code.strip()) for code in category_codes_str.split(",")
                ]

                LOGGER.warning(
                    f"Counting pixels for {len(category_codes)} manually specified classes: {category_codes}"
                )

            # Process each category code and add class counts to features
            # Start with the original feature collection
            result_fc = feature_collection

            # Process each category code sequentially
            for category_code in category_codes:
                # Create binary mask for this class
                specific_category = image.eq(category_code)

                # Count pixels of this class in each feature
                counts_per_grid = specific_category.reduceRegions(
                    collection=feature_collection,
                    reducer=ee.Reducer.sum(),
                    scale=self.scale,
                    tileScale=self.tile_scale,
                )

                # Join counts with result_fc by matching system:index
                # Create a lookup dictionary for faster access
                # Use aggregate_histogram to create a lookup
                count_lookup = counts_per_grid.aggregate_histogram("system:index")

                def add_class_count(feature):
                    """Add class count property to feature"""
                    # Get the feature's system:index
                    feature_index = feature.get("system:index")

                    # Find matching feature in counts_per_grid by system:index
                    matching_feature = counts_per_grid.filter(
                        ee.Filter.eq("system:index", feature_index)
                    ).first()

                    # Get the count value (default to 0 if not found)
                    class_count = ee.Algorithms.If(
                        matching_feature,
                        matching_feature.get("sum"),
                        ee.Number(0),
                    )

                    # Add class count property
                    return feature.set("class_" + str(category_code), class_count)

                # Add class count to all features
                result_fc = result_fc.map(add_class_count)

            LOGGER.warning(
                f"Successfully counted pixels for {len(category_codes)} classes across {feature_collection.size().getInfo()} features"
            )

            # Return Feature Collection with counts
            return knut.export_gee_connection(result_fc, fc_connection)

        except Exception as e:
            LOGGER.error(f"Count by class failed: {e}")
            raise
