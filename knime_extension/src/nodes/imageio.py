"""
GEE Image I/O Nodes for KNIME
This module contains nodes for reading, filtering, and processing Google Earth Engine Images and Image Collections.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

# Category for GEE Image I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="imageio",
    name="Image IO",
    description="Google Earth Engine Image Input/Output and Processing nodes",
    icon="icons/imageIO.png",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/image/"


############################################
# GEE Image Reader
############################################


@knext.node(
    name="Image Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "imagereader.png",
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
    name="Image Collection Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "icreader.png",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection with embedded collection object.",
    port_type=google_earth_engine_port_type,
)
class GEEImageCollectionReader:
    """Loads an image collection from Google Earth Engine.

    This node loads an Image Collection from GEE's catalog without applying any filters or aggregations.
    Use downstream filter and aggregator nodes to process the collection. This design provides maximum
    flexibility for building complex image processing workflows.

    **Common Collections:**

    - Sentinel-2: 'COPERNICUS/S2_SR' (10m resolution, optical)

    - Landsat 8: 'LANDSAT/LC08/C02/T1_L2' (30m resolution, optical)

    - MODIS: 'MODIS/006/MOD13Q1' (250m resolution, vegetation indices)

    - Landsat 7: 'LANDSAT/LE07/C02/T1_L2' (30m resolution, optical)

    - Sentinel-1: 'COPERNICUS/S1_GRD' (10m resolution, radar)

    **Workflow Design:**

    - Use this node to load the collection
    - Use **Image Collection Filter** for time, cloud, and property filtering
    - Use **Image Collection Spatial Filter** for spatial filtering and clipping
    - Use **Image Collection Aggregator** to create composite images
    """

    collection_id = knext.StringParameter(
        "Collection ID",
        "The ID of the GEE image collection (e.g., 'COPERNICUS/S2_SR')",
        default_value="COPERNICUS/S2_SR",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Load image collection
        image_collection = ee.ImageCollection(self.collection_id)

        LOGGER.warning(f"Loaded image collection: {self.collection_id}")

        return knut.export_gee_connection(image_collection, gee_connection)


############################################
# Image Collection Filter
############################################


@knext.node(
    name="Image Collection Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "icfilter.png",
    after="",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Filtered GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
class ImageCollectionFilter:
    """Filters an Image Collection by date and cloud cover.

    This node provides temporal and cloud cover filtering capabilities for Image Collections.
    Users must explicitly select the cloud property type (Sentinel-2, Landsat, or Custom)
    to ensure correct filtering. This is essential for creating clean, analysis-ready image composites.

    **Filter Types:**

    - **Date Filter**: Filter by start and end date (end date is inclusive)
    - **Cloud Filter**: Filter by cloud cover percentage using satellite-specific properties
    - **Sort & Limit**: Order results and limit number of images

    **Common Use Cases:**

    - Filter Sentinel-2 images by date and cloud cover
    - Get images within a specific time period
    - Limit collection to most recent N images
    - Sort images by acquisition date

    **Cloud Property Selection:**

    - **Sentinel-2**: Uses 'CLOUDY_PIXEL_PERCENTAGE' property
    - **Landsat**: Uses 'CLOUD_COVER' property
    - **Custom**: Specify your own cloud property name (e.g., 'CLOUD_COVER_LAND')

    **Important Notes:**

    - The end date is **inclusive** - selecting 2024-10-06 will include images from that entire day
    - For custom property filtering (e.g., orbit direction, tile ID), use the **Image Collection Value Filter** node
    - This node uses lazy evaluation for fast performance - the actual computation happens downstream
    """

    # Date filtering
    enable_date_filter = knext.BoolParameter(
        "Enable Date Filter",
        "Enable filtering by date range",
        default_value=True,
    )

    start_date = knext.DateTimeParameter(
        "Start Date",
        "Start date for filtering the image collection",
        default_value="2020-01-01",
        show_date=True,
        show_time=False,
    ).rule(knext.OneOf(enable_date_filter, [True]), knext.Effect.SHOW)

    end_date = knext.DateTimeParameter(
        "End Date",
        "End date for filtering the image collection",
        default_value="2024-12-31",
        show_date=True,
        show_time=False,
    ).rule(knext.OneOf(enable_date_filter, [True]), knext.Effect.SHOW)

    # Cloud filtering
    enable_cloud_filter = knext.BoolParameter(
        "Enable Cloud Filter",
        "Enable filtering by cloud cover percentage",
        default_value=False,
    )

    cloud_property_mode = knext.StringParameter(
        "Cloud Property Mode",
        "Select the cloud property name based on your satellite collection",
        default_value="Sentinel-2",
        enum=["Sentinel-2", "Landsat", "Custom"],
    ).rule(knext.OneOf(enable_cloud_filter, [True]), knext.Effect.SHOW)

    cloud_property_custom = knext.StringParameter(
        "Custom Cloud Property",
        "Custom cloud property name (only for Custom mode)",
        default_value="CLOUD_COVER",
    ).rule(
        knext.And(
            knext.OneOf(enable_cloud_filter, [True]),
            knext.OneOf(cloud_property_mode, ["Custom"]),
        ),
        knext.Effect.SHOW,
    )

    max_cloud_cover = knext.DoubleParameter(
        "Maximum Cloud Cover (%)",
        "Maximum cloud cover percentage (0-100)",
        default_value=20.0,
        min_value=0.0,
        max_value=100.0,
    ).rule(knext.OneOf(enable_cloud_filter, [True]), knext.Effect.SHOW)

    # Sort and limit
    enable_sort = knext.BoolParameter(
        "Enable Sorting",
        "Enable sorting by property",
        default_value=False,
    )

    sort_property = knext.StringParameter(
        "Sort Property",
        "Property to sort by (e.g., 'system:time_start' for chronological order)",
        default_value="system:time_start",
    ).rule(knext.OneOf(enable_sort, [True]), knext.Effect.SHOW)

    sort_ascending = knext.BoolParameter(
        "Sort Ascending",
        "Sort in ascending order (oldest first for dates)",
        default_value=True,
    ).rule(knext.OneOf(enable_sort, [True]), knext.Effect.SHOW)

    enable_limit = knext.BoolParameter(
        "Enable Limit",
        "Limit the number of images returned",
        default_value=False,
    )

    max_images = knext.IntParameter(
        "Maximum Images",
        "Maximum number of images to return",
        default_value=100,
        min_value=1,
        max_value=10000,
    ).rule(knext.OneOf(enable_limit, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging
        from datetime import datetime, timedelta, date

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.gee_object

        # Determine cloud property (user explicitly selects the type)
        cloud_property = None
        if self.enable_cloud_filter:
            if self.cloud_property_mode == "Sentinel-2":
                cloud_property = "CLOUDY_PIXEL_PERCENTAGE"
            elif self.cloud_property_mode == "Landsat":
                cloud_property = "CLOUD_COVER"
            elif self.cloud_property_mode == "Custom":
                cloud_property = self.cloud_property_custom

            LOGGER.warning(f"Using cloud property: {cloud_property}")

        # Apply date filter
        if self.enable_date_filter:
            s = self.start_date
            e = self.end_date

            # Helper function to convert to Python datetime
            def _to_py_dt(x):
                if x is None:
                    return None
                if hasattr(x, "to_pydatetime"):
                    return x.to_pydatetime()
                return x

            s = _to_py_dt(s)
            e = _to_py_dt(e)

            # Convert pure date objects to datetime
            if isinstance(s, date) and not isinstance(s, datetime):
                s = datetime.combine(s, datetime.min.time())
            if isinstance(e, date) and not isinstance(e, datetime):
                # Add 1 day to make the end date inclusive
                e = datetime.combine(e, datetime.min.time()) + timedelta(days=1)
            elif isinstance(e, datetime):
                # Add 1 day to make the end date inclusive
                e = e + timedelta(days=1)

            try:
                if s is not None and e is not None:
                    image_collection = image_collection.filterDate(
                        ee.Date(s), ee.Date(e)
                    )
                    LOGGER.warning(
                        f"Applied date filter: {s.date()} to {(e - timedelta(days=1)).date()} (inclusive)"
                    )
                elif s is not None:
                    image_collection = image_collection.filterDate(
                        ee.Date(s), ee.Date("2100-01-01")
                    )
                    LOGGER.warning(f"Applied date filter: >= {s.date()}")
                elif e is not None:
                    image_collection = image_collection.filterDate(
                        ee.Date("1900-01-01"), ee.Date(e)
                    )
                    LOGGER.warning(
                        f"Applied date filter: <= {(e - timedelta(days=1)).date()} (inclusive)"
                    )
                else:
                    LOGGER.warning(
                        "Date filter enabled but no start/end provided; skipping."
                    )
            except Exception as ex:
                LOGGER.warning(f"Date filter failed and was skipped: {ex}")

        # Apply cloud filter (cloud_property was determined at the beginning)
        if self.enable_cloud_filter:
            if cloud_property:
                try:
                    # Filter out images without the cloud property
                    image_collection = image_collection.filter(
                        ee.Filter.notNull([cloud_property])
                    )
                    # Apply cloud cover threshold
                    image_collection = image_collection.filter(
                        ee.Filter.lte(cloud_property, float(self.max_cloud_cover))
                    )
                    LOGGER.warning(
                        f"Applied cloud filter: {cloud_property} <= {self.max_cloud_cover}%"
                    )
                except Exception as ex:
                    LOGGER.warning(f"Cloud filter failed and was skipped: {ex}")
            else:
                LOGGER.warning(
                    "Cloud filter requested but no cloud property could be determined; skipped."
                )

        # Apply sort
        if self.enable_sort:
            try:
                image_collection = image_collection.sort(
                    self.sort_property, bool(self.sort_ascending)
                )
                LOGGER.warning(
                    f"Sorted by: {self.sort_property} ({'ascending' if self.sort_ascending else 'descending'})"
                )
            except Exception as ex:
                LOGGER.warning(f"Sort failed and was skipped: {ex}")

        # Apply limit
        if self.enable_limit:
            try:
                image_collection = image_collection.limit(int(self.max_images))
                LOGGER.warning(f"Limited to {self.max_images} images")
            except Exception as ex:
                LOGGER.warning(f"Limit failed and was skipped: {ex}")

        # Note: We skip checking result size here because image_collection.size().getInfo()
        # is an expensive operation that can be very slow for large collections.
        # The collection will be evaluated lazily when actually used downstream.

        return knut.export_gee_connection(image_collection, ic_connection)


############################################
# Image Collection Value Filter
############################################


@knext.node(
    name="Image Collection Value Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "icvaluefilter.png",
    after="",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Filtered GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
class ImageCollectionValueFilter:
    """Filters an Image Collection by custom metadata properties.

    This node provides flexible filtering capabilities based on image metadata properties
    such as orbit direction, tile ID, processing level, or any custom property available
    in the image collection. It supports multiple comparison operators for both numeric
    and string values.

    **Comparison Operators:**

    - **Equals**: Match exact values (supports both numeric and string)
    - **Not Equals**: Exclude exact values
    - **Greater Than**: Numeric comparison (>)
    - **Less Than**: Numeric comparison (<)
    - **Greater or Equal**: Numeric comparison (>=)
    - **Less or Equal**: Numeric comparison (<=)

    **Common Use Cases:**

    - Filter Sentinel-1 by orbit direction (ASCENDING/DESCENDING)
    - Select specific Sentinel-2 tiles by MGRS_TILE property
    - Filter by processing baseline or version
    - Select images from specific orbits or paths
    - Filter by custom metadata attributes

    **Common Properties:**

    - Sentinel-1: 'orbitProperties_pass' (ASCENDING/DESCENDING)
    - Sentinel-2: 'MGRS_TILE', 'SENSING_ORBIT_NUMBER'
    - Landsat: 'WRS_PATH', 'WRS_ROW', 'COLLECTION_NUMBER'
    - All: 'system:time_start', 'system:index'

    **Type Handling:**

    The node automatically handles both numeric and string property values.
    For numeric comparisons (>, <, >=, <=), the value will be parsed as a number.
    For Equals/Not Equals, the filter works with both numeric and string representations.
    """

    property_name = knext.StringParameter(
        "Property Name",
        "Name of the metadata property to filter by (e.g., 'orbitProperties_pass', 'MGRS_TILE')",
        default_value="",
    )

    property_operator = knext.StringParameter(
        "Comparison Operator",
        "Comparison operator for property filtering",
        default_value="Equals",
        enum=[
            "Equals",
            "Not Equals",
            "Greater Than",
            "Less Than",
            "Greater or Equal",
            "Less or Equal",
        ],
    )

    property_value = knext.StringParameter(
        "Property Value",
        "Value to compare against. For numeric comparisons, this will be converted to a number.",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.gee_object

        # Only apply filter if all parameters are provided
        if not self.property_name or not self.property_value:
            LOGGER.warning(
                "Property name or value is empty, returning original collection"
            )
            return knut.export_gee_connection(image_collection, ic_connection)

        prop_name = self.property_name.strip()
        val_str = self.property_value.strip()

        # Apply filter based on operator
        if self.property_operator in ["Equals", "Not Equals"]:
            # For Equals/Not Equals, use robust type handling (numeric + string)
            # Parse as number on server side
            num_filter = ee.Filter.eq(prop_name, ee.Number.parse(val_str))
            str_filter = ee.Filter.eq(prop_name, val_str)

            if self.property_operator == "Equals":
                # Match either numeric or string representation
                image_collection = image_collection.filter(
                    ee.Filter.Or(num_filter, str_filter)
                )
            else:  # Not Equals
                # Neither numeric nor string should match
                image_collection = image_collection.filter(
                    ee.Filter.And(
                        ee.Filter.neq(prop_name, ee.Number.parse(val_str)),
                        ee.Filter.neq(prop_name, val_str),
                    )
                )
        else:
            # For numeric comparisons, parse as number on server side
            numeric_value = ee.Number.parse(val_str)

            operator_map = {
                "Greater Than": ee.Filter.gt,
                "Less Than": ee.Filter.lt,
                "Greater or Equal": ee.Filter.gte,
                "Less or Equal": ee.Filter.lte,
            }

            filter_func = operator_map[self.property_operator]
            image_collection = image_collection.filter(
                filter_func(prop_name, numeric_value)
            )

        LOGGER.warning(
            f"Applied property filter: {prop_name} {self.property_operator} {val_str}"
        )

        # Check result size
        try:
            result_size = image_collection.size().getInfo()
            LOGGER.warning(f"Filtered collection contains {result_size} images")
            if result_size == 0:
                LOGGER.warning(
                    "Warning: Filter operation resulted in empty Image Collection"
                )
        except Exception as e:
            LOGGER.warning(f"Could not check result size: {e}")

        return knut.export_gee_connection(image_collection, ic_connection)


############################################
# Image Collection Spatial Filter
############################################


@knext.node(
    name="Image Collection Spatial Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "icspatialfilter.png",
    after="",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="ROI Feature Collection Connection",
    description="Feature Collection defining the region of interest.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Spatially filtered GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
class ImageCollectionSpatialFilter:
    """Filters and clips an Image Collection using a region of interest (ROI).

    This node provides spatial filtering and clipping capabilities for Image Collections.
    It can filter images that intersect with an ROI and optionally clip each image to the ROI boundary.
    This is essential for focusing analysis on specific geographic areas.

    **Spatial Operations:**

    - **Filter by Bounds**: Keep only images that intersect the ROI
    - **Clip to ROI**: Clip each image to the exact ROI boundary
    - **Combined**: Filter by bounds and clip simultaneously

    **Common Use Cases:**

    - Extract images for a specific study area
    - Clip satellite imagery to administrative boundaries
    - Focus analysis on specific regions to reduce processing time
    - Prepare data for area-specific analysis

    **Performance Notes:**

    - **filterBounds** is fast and efficient for spatial filtering
    - **Clipping** adds processing time but provides precise boundaries
    - For large ROIs, consider using filterBounds only
    """

    filter_bounds = knext.BoolParameter(
        "Filter by Bounds",
        "Filter images to only those intersecting the ROI",
        default_value=True,
    )

    clip_to_roi = knext.BoolParameter(
        "Clip to ROI",
        "Clip each image to the ROI boundary",
        default_value=False,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
        roi_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.gee_object
        roi_geometry = roi_connection.gee_object.geometry()

        # Apply spatial filter
        if self.filter_bounds:
            image_collection = image_collection.filterBounds(roi_geometry)
            LOGGER.warning("Applied filterBounds to Image Collection")

        # Apply clipping
        if self.clip_to_roi:

            def clip_image(image):
                return image.clip(roi_geometry)

            image_collection = image_collection.map(clip_image)
            LOGGER.warning("Applied clip to each image in collection")

        # Check result size
        try:
            result_size = image_collection.size().getInfo()
            LOGGER.warning(
                f"Spatially filtered collection contains {result_size} images"
            )
            if result_size == 0:
                LOGGER.warning(
                    "Warning: Spatial filter resulted in empty Image Collection"
                )
        except Exception as e:
            LOGGER.warning(f"Could not check result size: {e}")

        return knut.export_gee_connection(image_collection, ic_connection)


############################################
# Image Collection Aggregator
############################################


@knext.node(
    name="Image Collection Aggregator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "icaggregator.png",
    after="",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with aggregated image.",
    port_type=google_earth_engine_port_type,
)
class ImageCollectionAggregator:
    """Aggregates an Image Collection into a single composite image.

    This node reduces an Image Collection to a single image by applying various aggregation
    methods. This is essential for creating cloud-free composites, temporal averages,
    and other statistical summaries of image time series.

    **Aggregation Methods:**

    - **first**: Returns the first image (useful for already-filtered collections)
    - **last**: Returns the most recent image
    - **mean**: Calculates pixel-wise mean (good for reducing noise)
    - **median**: Calculates pixel-wise median (robust to outliers, best for cloud removal)
    - **min**: Finds minimum values (useful for NDVI minimum)
    - **max**: Finds maximum values (useful for NDVI maximum)
    - **sum**: Adds pixel values (useful for accumulation)
    - **mode**: Finds most frequent values (useful for classification)
    - **mosaic**: Creates a mosaic (first valid pixel)

    **Common Use Cases:**

    - Create cloud-free composites using median
    - Calculate temporal averages for change detection
    - Find maximum NDVI values over a growing season
    - Generate annual precipitation sums
    - Create mosaics from overlapping images

    **Best Practices:**

    - Use **median** for cloud removal in optical imagery
    - Use **mean** for temporal averaging
    - Use **mosaic** for seamless image mosaicking
    - Apply filters before aggregation to improve results
    """

    aggregation_method = knext.StringParameter(
        "Aggregation Method",
        "Method to aggregate multiple images into one",
        default_value="median",
        enum=["first", "last", "mean", "median", "min", "max", "sum", "mode", "mosaic"],
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.gee_object

        # Define aggregation methods
        aggregation_methods = {
            "first": lambda ic: ic.first(),
            "last": lambda ic: ic.sort("system:time_start", False).first(),
            "mean": lambda ic: ic.mean(),
            "median": lambda ic: ic.median(),
            "min": lambda ic: ic.min(),
            "max": lambda ic: ic.max(),
            "sum": lambda ic: ic.sum(),
            "mode": lambda ic: ic.mode(),
            "mosaic": lambda ic: ic.mosaic(),
        }

        # Apply aggregation
        try:
            image = aggregation_methods[self.aggregation_method](image_collection)
            LOGGER.warning(
                f"Successfully aggregated using '{self.aggregation_method}' method"
            )
        except Exception as e:
            LOGGER.error(
                f"Aggregation method '{self.aggregation_method}' failed: {e}. Falling back to 'first'."
            )
            image = image_collection.first()

        return knut.export_gee_connection(image, ic_connection)


############################################
# Image Band Selector
############################################


@knext.node(
    name="Image Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "bandselector.png",
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

        # Only process if bands parameter is not empty
        if self.bands.strip():
            # Get original image info for logging

            # Optimize: only get band names, not full image info
            original_bands = image.bandNames().getInfo()
            # LOGGER.warning(f"Original image bands: {original_bands}")

            band_list = [band.strip() for band in self.bands.split(",") if band.strip()]
        # LOGGER.warning(f"Selecting bands: {band_list}")

        # Filter to only include bands that exist in the image
        available_bands = [band for band in band_list if band in original_bands]
        if available_bands:
            image = image.select(available_bands)
        else:
            LOGGER.warning(
                f"No specified bands found in image. Available bands: {original_bands}"
            )

        return knut.export_gee_connection(image, image_connection)


############################################
# Get Image Value by LatLon
############################################


@knext.node(
    name="Get Image Value by LatLon",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "getvalue.png",
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
    icon_path=__NODE_ICON_PATH + "reducer.png",
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
