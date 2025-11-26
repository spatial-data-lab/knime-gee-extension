"""
GEE Image I/O Nodes for KNIME
This module contains nodes for reading, filtering, and processing Google Earth Engine Image Collections.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
    gee_image_collection_port_type,
    gee_feature_collection_port_type,
)

# Category for GEE Image I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="imagecollection",
    name="Image Collection IO",
    description="Google Earth Engine Image Input/Output and Processing nodes",
    icon="icons/ImageCollection.png",
    after="authorization",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/imagecollection/"


############################################
# Dataset Search
############################################


@knext.node(
    name="GEE Dataset Search",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "DatasetSearch.png",
    id="datasetsearch",
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
        """The source to search from GEE data catalog:
        
        - 'ee': Official Google Earth Engine datasets
        - 'community': Community-contributed datasets
        - 'all': Both official and community datasets
        """,
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
# GEE Image Collection Reader
############################################


@knext.node(
    name="Image Collection Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionReader.png",
    id="imagecollectionreader",
    after="datasetsearch",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection with embedded collection object.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionReader:
    """Loads an image collection from Google Earth Engine.

    This node loads an Image Collection from GEE's catalog without applying any filters or aggregations.
    Use downstream filter and aggregator nodes to process the collection. This design provides maximum
    flexibility for building complex image processing workflows.

    **Common Collections:**

    - [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED): 'COPERNICUS/S2_SR' (10m resolution, optical)
    - [Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2): 'LANDSAT/LC08/C02/T1_L2' (30m resolution, optical)
    - [MODIS](https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD13Q1): 'MODIS/006/MOD13Q1' (250m resolution, vegetation indices)
    - [Landsat 7](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2): 'LANDSAT/LE07/C02/T1_L2' (30m resolution, optical)
    - [Sentinel-1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD): 'COPERNICUS/S1_GRD' (10m resolution, radar)
    **Workflow Design:**

    - Use this node to load the collection
    - Use **Image Collection Filter** for time, cloud, and property filtering
    - Use **Image Collection Spatial Filter** for spatial filtering and clipping
    - Use **Image Collection Aggregator** to create composite images

    **Note:** The GEE Dataset Search node can help you find more image collections.
    """

    collection_id = knext.StringParameter(
        "Collection ID",
        """The ID of the GEE image collection (e.g., 'COPERNICUS/S2_SR'). 
        You can use the GEE Dataset Search node to find available collections or 
        visit [GEE Datasets Catalog](https://developers.google.com/earth-engine/datasets/catalog).""",
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

        return knut.export_gee_image_collection_connection(
            image_collection, gee_connection
        )


############################################
# Image Collection General Filter
############################################


@knext.node(
    name="Image Collection General Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionFilter.png",
    id="imagecollectionfilter",
    after="imagecollectionreader",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Filtered GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
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
        image_collection = ic_connection.image_collection

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

        return knut.export_gee_image_collection_connection(
            image_collection, ic_connection
        )


############################################
# Image Collection Value Filter
############################################


# @knext.node(
#     name="Image Collection Value Filter",
#     node_type=knext.NodeType.MANIPULATOR,
#     category=__category,
#     icon_path=__NODE_ICON_PATH + "ImageCollectionValueFilter.png",
#     id="imagecollectionvaluefilter",
#     after="imagecollectionfilter",
# )
# @knext.input_port(
#     name="GEE Image Collection Connection",
#     description="GEE Image Collection connection.",
#     port_type=google_earth_engine_port_type,
# )
# @knext.output_port(
#     name="GEE Image Collection Connection",
#     description="Filtered GEE Image Collection connection.",
#     port_type=google_earth_engine_port_type,
# )
# class ImageCollectionValueFilter:
#     """Filters an Image Collection by custom metadata properties.

#     This node provides flexible filtering capabilities based on image metadata properties
#     such as orbit direction, tile ID, processing level, or any custom property available
#     in the image collection. It supports multiple comparison operators for both numeric
#     and string values.

#     **Comparison Operators:**

#     - **Equals**: Match exact values (supports both numeric and string)
#     - **Not Equals**: Exclude exact values
#     - **Greater Than**: Numeric comparison (>)
#     - **Less Than**: Numeric comparison (<)
#     - **Greater or Equal**: Numeric comparison (>=)
#     - **Less or Equal**: Numeric comparison (<=)

#     **Common Use Cases:**

#     - Filter Sentinel-1 by orbit direction (ASCENDING/DESCENDING)
#     - Select specific Sentinel-2 tiles by MGRS_TILE property
#     - Filter by processing baseline or version
#     - Select images from specific orbits or paths
#     - Filter by custom metadata attributes

#     **Common Properties:**

#     - Sentinel-1: 'orbitProperties_pass' (ASCENDING/DESCENDING)
#     - Sentinel-2: 'MGRS_TILE', 'SENSING_ORBIT_NUMBER'
#     - Landsat: 'WRS_PATH', 'WRS_ROW', 'COLLECTION_NUMBER'
#     - All: 'system:time_start', 'system:index'

#     **Type Handling:**

#     The node automatically handles both numeric and string property values.
#     For numeric comparisons (>, <, >=, <=), the value will be parsed as a number.
#     For Equals/Not Equals, the filter works with both numeric and string representations.
#     """

#     property_name = knext.StringParameter(
#         "Property Name",
#         "Name of the metadata property to filter by (e.g., 'orbitProperties_pass', 'MGRS_TILE')",
#         default_value="",
#     )

#     property_operator = knext.StringParameter(
#         "Comparison Operator",
#         "Comparison operator for property filtering",
#         default_value="Equals",
#         enum=[
#             "Equals",
#             "Not Equals",
#             "Greater Than",
#             "Less Than",
#             "Greater or Equal",
#             "Less or Equal",
#         ],
#     )

#     property_value = knext.StringParameter(
#         "Property Value",
#         "Value to compare against. For numeric comparisons, this will be converted to a number.",
#         default_value="",
#     )

#     def configure(self, configure_context, input_schema):
#         return None

#     def execute(self, exec_context: knext.ExecutionContext, ic_connection):
#         import ee
#         import logging

#         LOGGER = logging.getLogger(__name__)
#         image_collection = ic_connection.image_collection

#         # Only apply filter if all parameters are provided
#         if not self.property_name or not self.property_value:
#             LOGGER.warning(
#                 "Property name or value is empty, returning original collection"
#             )
#             return knut.export_gee_image_collection_connection(image_collection, ic_connection)

#         prop_name = self.property_name.strip()
#         val_str = self.property_value.strip()

#         # 【关键修正】Client-side check to determine if the value is numeric
#         is_numeric = False
#         numeric_value = None
#         try:
#             numeric_value = float(val_str)
#             # Avoid NaN/Inf
#             if (
#                 numeric_value == float("inf")
#                 or numeric_value == float("-inf")
#                 or numeric_value != numeric_value
#             ):
#                 is_numeric = False
#             else:
#                 is_numeric = True
#         except ValueError:
#             is_numeric = False

#         the_filter = None

#         # Apply filter based on operator
#         if self.property_operator == "Equals":
#             # If numeric, compare as a number; otherwise, as a string.
#             if is_numeric:
#                 the_filter = ee.Filter.eq(prop_name, numeric_value)
#             else:
#                 the_filter = ee.Filter.eq(prop_name, val_str)

#         elif self.property_operator == "Not Equals":
#             if is_numeric:
#                 the_filter = ee.Filter.neq(prop_name, numeric_value)
#             else:
#                 the_filter = ee.Filter.neq(prop_name, val_str)

#         else:  # All other operators are numeric
#             if not is_numeric:
#                 raise ValueError(
#                     f"Operator '{self.property_operator}' requires a numeric Property Value, but got '{val_str}'."
#                 )

#             operator_map = {
#                 "Greater Than": ee.Filter.gt,
#                 "Less Than": ee.Filter.lt,
#                 "Greater or Equal": ee.Filter.gte,
#                 "Less or Equal": ee.Filter.lte,
#             }
#             filter_func = operator_map[self.property_operator]
#             the_filter = filter_func(prop_name, numeric_value)

#         # Apply the created filter to the collection
#         image_collection = image_collection.filter(the_filter)

#         LOGGER.warning(
#             f"Applied property filter: {prop_name} {self.property_operator} {val_str}"
#         )

#         # Check result size
#         try:
#             result_size = image_collection.size().getInfo()
#             LOGGER.warning(f"Filtered collection contains {result_size} images")
#             if result_size == 0:
#                 LOGGER.warning(
#                     "Warning: Filter operation resulted in empty Image Collection"
#                 )
#         except Exception as e:
#             LOGGER.warning(f"Could not check result size: {e}")

#         return knut.export_gee_image_collection_connection(image_collection, ic_connection)


############################################
# Image Collection Spatial Filter
############################################


@knext.node(
    name="Image Collection Spatial Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionSpatialFilter.png",
    id="imagecollectionspatialfilter",
    after="imagecollectionfilter",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
)
@knext.input_port(
    name="ROI Feature Collection Connection",
    description="Feature Collection defining the region of interest.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Spatially filtered GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
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
        "Clip each image to the ROI boundary. Note: This may increase processing time.",
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
        image_collection = ic_connection.image_collection
        roi_geometry = roi_connection.feature_collection.geometry()

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

        return knut.export_gee_image_collection_connection(
            image_collection, ic_connection
        )


############################################
# Image Collection Aggregator
############################################


@knext.node(
    name="Image Collection Aggregator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionAggregator.png",
    id="imagecollectionaggregator",
    after="imagecollectionspatialfilter",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with aggregated image.",
    port_type=gee_image_port_type,
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
        """Method to aggregate multiple images into one. 
        Available methods:
        
        - **first**: Returns the first image (useful for already-filtered collections)
        - **last**: Returns the most recent image
        - **mean**: Calculates pixel-wise mean (good for reducing noise)
        - **median**: Calculates pixel-wise median (robust to outliers, best for cloud removal)
        - **min**: Finds minimum values (useful for NDVI minimum)
        - **max**: Finds maximum values (useful for NDVI maximum)
        - **sum**: Adds pixel values (useful for accumulation)
        - **mode**: Finds most frequent values (useful for classification)
        - **mosaic**: Creates a mosaic (first valid pixel)
        """,
        default_value="median",
        enum=["first", "last", "mean", "median", "min", "max", "sum", "mode", "mosaic"],
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.image_collection

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

        return knut.export_gee_image_connection(image, ic_connection)
