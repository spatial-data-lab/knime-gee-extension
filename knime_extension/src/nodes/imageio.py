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
    icon="icons/ImageIO.png",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/image/"


############################################
# GEE Image Collection Reader
############################################


@knext.node(
    name="Image Collection Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionReader.png",
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
class ImageCollectionReader:
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
    name="Image Collection General Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionFilter.png",
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
class ImageCollectionGeneralFilter:
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
    icon_path=__NODE_ICON_PATH + "ImageCollectionValueFilter.png",
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

        # 【关键修正】Client-side check to determine if the value is numeric
        is_numeric = False
        numeric_value = None
        try:
            numeric_value = float(val_str)
            # Avoid NaN/Inf
            if (
                numeric_value == float("inf")
                or numeric_value == float("-inf")
                or numeric_value != numeric_value
            ):
                is_numeric = False
            else:
                is_numeric = True
        except ValueError:
            is_numeric = False

        the_filter = None

        # Apply filter based on operator
        if self.property_operator == "Equals":
            # If numeric, compare as a number; otherwise, as a string.
            if is_numeric:
                the_filter = ee.Filter.eq(prop_name, numeric_value)
            else:
                the_filter = ee.Filter.eq(prop_name, val_str)

        elif self.property_operator == "Not Equals":
            if is_numeric:
                the_filter = ee.Filter.neq(prop_name, numeric_value)
            else:
                the_filter = ee.Filter.neq(prop_name, val_str)

        else:  # All other operators are numeric
            if not is_numeric:
                raise ValueError(
                    f"Operator '{self.property_operator}' requires a numeric Property Value, but got '{val_str}'."
                )

            operator_map = {
                "Greater Than": ee.Filter.gt,
                "Less Than": ee.Filter.lt,
                "Greater or Equal": ee.Filter.gte,
                "Less or Equal": ee.Filter.lte,
            }
            filter_func = operator_map[self.property_operator]
            the_filter = filter_func(prop_name, numeric_value)

        # Apply the created filter to the collection
        image_collection = image_collection.filter(the_filter)

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
    icon_path=__NODE_ICON_PATH + "ImageCollectionSpatialFilter.png",
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
    icon_path=__NODE_ICON_PATH + "ImageCollectionAggregator.png",
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
# GEE Image Reader
############################################


@knext.node(
    name="Image Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageReader.png",
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
class ImageReader:
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
# Image Band Selector
############################################


@knext.node(
    name="Image Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandSelector.png",
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
class ImageBandSelector:
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
# Dataset Search
############################################


@knext.node(
    name="GEE Dataset Search",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "DatasetSearch.png",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_table(
    name="Search Results",
    description="Table containing search results from GEE data catalog",
)
class GEEDatasetSearch:
    """Searches datasets from Google Earth Engine data catalog.

    This node searches the Google Earth Engine data catalog for datasets matching your criteria.
    It provides a powerful way to discover available satellite imagery, elevation data, and other
    geospatial datasets in GEE's extensive catalog.

    **Search Options:**

    - **Keyword Search**: Search by dataset name, description, or tags
    - **Source Filter**: Search in official GEE datasets, community datasets, or both
    - **Regex Support**: Use regular expressions for advanced pattern matching

    **Common Use Cases:**

    - Discover available satellite imagery for your study area
    - Find elevation or land cover datasets
    - Search for specific sensor data (Sentinel, Landsat, MODIS, etc.)
    - Explore community-contributed datasets

    **Search Tips:**

    - Use specific sensor names: "Sentinel-2", "Landsat", "MODIS"
    - Search by data type: "elevation", "landcover", "precipitation"
    - Use geographic terms: "global", "US", "Europe"
    - Enable regex for pattern matching: "S2.*SR" for Sentinel-2 Surface Reflectance
    """

    search_keyword = knext.StringParameter(
        "Search Keyword",
        "The keyword to search from GEE data catalog (e.g., 'Sentinel-2', 'elevation', 'SRTM')",
        default_value="Sentinel-2",
    )

    source = knext.StringParameter(
        "Source",
        "The source to search from GEE data catalog",
        default_value="ee",
        enum=["ee", "community", "all"],
    )

    use_regex = knext.BoolParameter(
        "Use Regular Expression",
        "Use regular expression for advanced pattern matching",
        default_value=False,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import pandas as pd
        from geemap import common as cm
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Search GEE data catalog
            search_result = cm.search_ee_data(
                self.search_keyword, regex=self.use_regex, source=self.source
            )

            if search_result:
                df = pd.DataFrame(search_result)
                LOGGER.warning(
                    f"Found {len(df)} datasets matching '{self.search_keyword}'"
                )
            else:
                # Return empty DataFrame with expected columns if no results
                df = pd.DataFrame(
                    columns=[
                        "id",
                        "title",
                        "provider",
                        "tags",
                        "start_date",
                        "end_date",
                    ]
                )
                LOGGER.warning(f"No datasets found matching '{self.search_keyword}'")

            return knext.Table.from_pandas(df)

        except Exception as e:
            LOGGER.error(f"Dataset search failed: {e}")
            # Return empty DataFrame on error
            df = pd.DataFrame(
                columns=["id", "title", "provider", "tags", "start_date", "end_date"]
            )
            return knext.Table.from_pandas(df)


############################################
# Image Get Info
############################################


@knext.node(
    name="Image Get Info",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageGetInfo.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection (pass-through).",
    port_type=google_earth_engine_port_type,
)
@knext.output_view(
    name="Image Info View",
    description="HTML view showing detailed image information",
)
class ImageGetInfo:
    """Displays detailed information about a Google Earth Engine image.

    This node extracts and displays comprehensive metadata about a GEE image including
    band information, properties, geometry, and other technical details. This is useful
    for understanding image structure before processing, verifying band names and properties,
    and debugging image-related issues.

    **Information Displayed:**

    - **Band Information**: Names, types, and properties of all bands
    - **Image Properties**: Metadata, acquisition date, cloud cover, etc.
    - **Geometry**: Bounding box and projection information
    - **System Properties**: GEE internal properties and identifiers

    **Use Cases:**

    - Explore image structure and available bands
    - Verify image properties before analysis
    - Debug image processing issues
    - Understand image metadata and acquisition parameters
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import json
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image from connection (GEE already initialized)
            image = image_connection.gee_object

            # Get image information efficiently
            info = image.getInfo()

            # Format as JSON for display
            json_string = json.dumps(info, indent=2)

            # Create HTML view
            html = f"""
            <div style="font-family: monospace; font-size: 12px;">
                <h3>Image Information</h3>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
{json_string}
                </pre>
            </div>
            """

            LOGGER.warning("Successfully retrieved image information")
            return knut.export_gee_connection(image, image_connection), knext.view_html(
                html
            )

        except Exception as e:
            LOGGER.error(f"Failed to get image info: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Error</h3>
                <p>Failed to retrieve image information: {str(e)}</p>
            </div>
            """
            return knut.export_gee_connection(image, image_connection), knext.view_html(
                error_html
            )


############################################
# Image Exporter
############################################


@knext.node(
    name="Image Exporter",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ExportImage.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
class ImageExporter:
    """Exports a Google Earth Engine image to local storage.

    This node exports a GEE image to your local file system in various formats.
    It handles the export process efficiently and provides options for controlling
    the export resolution and file format.

    **Export Options:**

    - **Output Path**: Specify the local file path for the exported image
    - **Scale**: Control the export resolution (meters per pixel)
    - **Format**: Automatically determined by file extension

    **Supported Formats:**

    - **GeoTIFF**: .tif, .tiff (recommended for most use cases)
    - **JPEG**: .jpg, .jpeg (for visualization)
    - **PNG**: .png (for visualization)

    **Use Cases:**

    - Export processed images for further analysis in other software
    - Create high-resolution maps for presentations
    - Generate base maps for GIS applications
    - Export classification results or derived products

    **Performance Notes:**

    - Larger scale values (lower resolution) export faster
    - Very high resolution exports may take considerable time
    - Consider using appropriate scale for your analysis needs
    """

    output_path = knext.StringParameter(
        "Output Path",
        "Local file path for the exported image (include file extension, e.g., 'output.tif')",
        default_value="exported_image.tif",
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters per pixel for export (lower = higher resolution)",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import geemap
        import logging
        import os

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image from connection (GEE already initialized)
            image = image_connection.gee_object

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Export image using geemap
            geemap.ee_export_image(
                image,
                filename=self.output_path,
                scale=self.scale,
                region=image.geometry(),
            )

            LOGGER.warning(f"Successfully exported image to: {self.output_path}")
            LOGGER.warning(f"Export scale: {self.scale} meters per pixel")

        except Exception as e:
            LOGGER.error(f"Image export failed: {e}")
            raise


############################################
# GeoTiff To GEE Image
############################################


@knext.node(
    name="Local GeoTiff To GEE",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GeotiffToGEE.png",
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
@knext.output_view(
    name="Image Info View",
    description="HTML view showing imported image information",
)
class GeoTiffToGEEImage:
    """Converts a local GeoTIFF file to a Google Earth Engine image.

    This node uploads and converts a local GeoTIFF file to a GEE image object,
    making it available for processing within GEE workflows. This is useful for
    incorporating local data, custom analysis results, or external datasets into
    GEE-based analysis.

    **Supported Formats:**

    - **GeoTIFF**: .tif, .tiff files with geographic information
    - **Single Band**: Grayscale or single-band images
    - **Multi-Band**: RGB, multispectral, or hyperspectral images

    **Use Cases:**

    - Upload custom classification results for further analysis
    - Incorporate local survey data or field measurements
    - Import external satellite imagery not available in GEE
    - Upload processed results from other software for visualization

    **Important Notes:**

    - File must be in a supported geographic projection (preferably WGS84)
    - Large files may take time to upload and process
    - Ensure sufficient storage quota in your GEE account
    - File path must be accessible from the KNIME environment
    """

    local_tiff_path = knext.StringParameter(
        "Local GeoTIFF Path",
        "Full path to the local GeoTIFF file to upload",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import geemap
        import logging
        import json
        import os

        LOGGER = logging.getLogger(__name__)

        try:
            # Validate file path
            if not self.local_tiff_path or not os.path.exists(self.local_tiff_path):
                raise FileNotFoundError(
                    f"GeoTIFF file not found: {self.local_tiff_path}"
                )

            # Convert local GeoTIFF to GEE image
            image = geemap.tif_to_ee(self.local_tiff_path)

            # Get basic image information for display
            info = image.getInfo()
            json_string = json.dumps(info, indent=2)

            # Create HTML view
            html = f"""
            <div style="font-family: monospace; font-size: 12px;">
                <h3>Imported Image Information</h3>
                <p><strong>File:</strong> {os.path.basename(self.local_tiff_path)}</p>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
{json_string}
                </pre>
            </div>
            """

            LOGGER.warning(f"Successfully imported GeoTIFF: {self.local_tiff_path}")
            return knut.export_gee_connection(image, gee_connection), knext.view_html(
                html
            )

        except Exception as e:
            LOGGER.error(f"GeoTIFF import failed: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Import Error</h3>
                <p>Failed to import GeoTIFF: {str(e)}</p>
                <p><strong>File:</strong> {self.local_tiff_path}</p>
            </div>
            """
            # Return a dummy image connection to maintain workflow continuity
            dummy_image = ee.Image.constant(0)
            return knut.export_gee_connection(
                dummy_image, gee_connection
            ), knext.view_html(error_html)
