"""
GEE Image Collection nodes for KNIME.
Image collection I/O and processing: search, filter, composite, time series, map operations.
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
    name="GEE Image Collection",
    description="Image collection I/O and processing: search, filter, composite, time series, map operations.",
    icon="icons/ImageCollection.png",
    after="featureio",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/imagecollection/"


############################################
# Dataset Search
############################################


@knext.node(
    name="GEE Dataset Searcher",
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

    This node searches the Google Earth Engine data catalog using Search keyword, Source, and Regex to control matching, and is commonly used to discover datasets before building workflows.

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
        "Search keyword",
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
        "Use regular expression",
        """Use regular expression for advanced pattern matching. 
        For more details about regular expression, 
        see the [Wikipedia article](https://en.wikipedia.org/wiki/Regular_expression).""",
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
    name="GEE Image Collection Reader",
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

    This node loads a Google Earth Engine Image Collection using Collection ID to select the dataset, and is commonly used as the starting point for filtering and compositing workflows.

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

        collection_id = (self.collection_id or "").strip()
        if not collection_id:
            raise ValueError(
                "Collection ID cannot be empty. Please provide a valid GEE image collection ID."
            )

        try:
            asset = ee.data.getAsset(collection_id)
        except Exception as e:
            LOGGER.error(f"Failed to fetch asset metadata for '{collection_id}': {e}")
            raise ValueError(
                f"Asset '{collection_id}' was not found or is not accessible. "
                f"Please verify the collection ID and your access permissions. Error: {str(e)}"
            ) from e

        asset_type = (asset or {}).get("type")
        if asset_type != "IMAGE_COLLECTION":
            suggested = {
                "IMAGE": "GEE Image Reader",
                "FEATURE_COLLECTION": "GEE Feature Collection Reader",
            }.get(asset_type, "the appropriate reader node")
            raise ValueError(
                f"Asset '{collection_id}' is not an Image Collection (type: {asset_type}). "
                f"Please use {suggested}."
            )

        # Load image collection
        image_collection = ee.ImageCollection(collection_id)

        LOGGER.warning(f"Loaded image collection: {collection_id}")

        return knut.export_gee_image_collection_connection(
            image_collection, gee_connection
        )


############################################
# Image Collection Info Extractor
############################################


@knext.node(
    name="GEE Image Collection Info Extractor",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICInfo.png",
    id="imagecollectioninfo",
    after="imagecollectionreader",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection to inspect.",
    port_type=gee_image_collection_port_type,
)
@knext.output_table(
    name="Image Collection Info Table",
    description="Table with property and value columns: count, date_min, date_max, band_names (comma-separated).",
)
class ImageCollectionGetInfo:
    """Outputs a two-column summary table of Image Collection properties.

    This node outputs a small summary table (count, date_min, date_max, band_names) and is commonly used to validate a collection before further processing.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
    ):
        import ee
        import logging
        import pandas as pd
        from datetime import datetime

        LOGGER = logging.getLogger(__name__)

        try:
            ic = ic_connection.image_collection

            count = ic.size().getInfo()
            band_names_list = ic.first().bandNames().getInfo()
            band_names_str = (
                ",".join(str(b) for b in band_names_list) if band_names_list else ""
            )

            try:
                t_min_ms = ic.aggregate_min("system:time_start").getInfo()
                t_max_ms = ic.aggregate_max("system:time_start").getInfo()
                date_min = (
                    datetime.fromtimestamp(t_min_ms / 1000).strftime("%Y-%m-%d")
                    if t_min_ms is not None
                    else ""
                )
                date_max = (
                    datetime.fromtimestamp(t_max_ms / 1000).strftime("%Y-%m-%d")
                    if t_max_ms is not None
                    else ""
                )
            except Exception as e:
                LOGGER.warning("Could not get date range: %s", e)
                date_min = ""
                date_max = ""

            rows = [
                {"property": "count", "value": str(count)},
                {"property": "date_min", "value": date_min or ""},
                {"property": "date_max", "value": date_max or ""},
                {"property": "band_names", "value": band_names_str},
            ]
            df = pd.DataFrame(rows)
            LOGGER.warning(
                "Image Collection Info: count=%s, date_min=%s, date_max=%s",
                count,
                date_min,
                date_max,
            )
            return knext.Table.from_pandas(df)

        except Exception as e:
            LOGGER.error("Failed to get Image Collection info: %s", e)
            df = pd.DataFrame([{"property": "error", "value": str(e)}])
            return knext.Table.from_pandas(df)


############################################
# Image Collection General Filter
############################################


@knext.node(
    name="GEE Image Collection General Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionFilter.png",
    id="imagecollectionfilter",
    after="imagecollectioninfo",
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

    This node filters an Image Collection using date range, cloud threshold, sort, and limit parameters, and is commonly used to clean collections before compositing.

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
        "Enable date filter",
        "Enable filtering by date range",
        default_value=True,
    )

    start_date = knext.DateTimeParameter(
        "Start date",
        "Start date for filtering the image collection",
        default_value="2020-01-01",
        show_date=True,
        show_time=False,
    ).rule(knext.OneOf(enable_date_filter, [True]), knext.Effect.SHOW)

    end_date = knext.DateTimeParameter(
        "End date",
        "End date for filtering the image collection",
        default_value="2024-12-31",
        show_date=True,
        show_time=False,
    ).rule(knext.OneOf(enable_date_filter, [True]), knext.Effect.SHOW)

    # Cloud filtering
    enable_cloud_filter = knext.BoolParameter(
        "Enable cloud filter",
        "Enable filtering by cloud cover percentage",
        default_value=False,
    )

    cloud_property_mode = knext.StringParameter(
        "Cloud property mode",
        "Select the cloud property name based on your satellite collection",
        default_value="Sentinel-2",
        enum=["Sentinel-2", "Landsat", "Custom"],
    ).rule(knext.OneOf(enable_cloud_filter, [True]), knext.Effect.SHOW)

    cloud_property_custom = knext.StringParameter(
        "Custom cloud property",
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
        "Maximum cloud cover (%)",
        "Maximum cloud cover percentage (0-100)",
        default_value=20.0,
        min_value=0.0,
        max_value=100.0,
    ).rule(knext.OneOf(enable_cloud_filter, [True]), knext.Effect.SHOW)

    # Sort and limit
    enable_sort = knext.BoolParameter(
        "Enable sorting",
        "Enable sorting by property",
        default_value=False,
    )

    sort_property = knext.StringParameter(
        "Sort property",
        "Property to sort by (e.g., 'system:time_start' for chronological order)",
        default_value="system:time_start",
    ).rule(knext.OneOf(enable_sort, [True]), knext.Effect.SHOW)

    sort_ascending = knext.BoolParameter(
        "Sort ascending",
        "Sort in ascending order (oldest first for dates)",
        default_value=True,
    ).rule(knext.OneOf(enable_sort, [True]), knext.Effect.SHOW)

    enable_limit = knext.BoolParameter(
        "Enable limit",
        "Limit the number of images returned",
        default_value=False,
    )

    max_images = knext.IntParameter(
        "Maximum images",
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

            # Convert pure date objects to datetime (end date is exclusive: last day not included)
            if isinstance(s, date) and not isinstance(s, datetime):
                s = datetime.combine(s, datetime.min.time())
            if isinstance(e, date) and not isinstance(e, datetime):
                e = datetime.combine(e, datetime.min.time())
            # e is used as exclusive end: filterDate(s, e) => [s, e), so e is not included

            try:
                if s is not None and e is not None:
                    image_collection = image_collection.filterDate(
                        ee.Date(s), ee.Date(e)
                    )
                    LOGGER.warning(
                        f"Applied date filter: {s.date()} to {e.date()} (end exclusive)"
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
                    LOGGER.warning(f"Applied date filter: < {e.date()} (end exclusive)")
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
# Image Collection Value Filter (by metadata)
############################################


@knext.node(
    name="GEE Image Collection Value Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionFilter.png",
    id="imagecollectionvaluefilter",
    after="imagecollectionspatialfilter",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Filtered or per-image masked GEE Image Collection.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionValueFilter:
    __doc__ = (
        "Filter by image metadata or mask by band value using updateMask per image.\n\n"
        "This node filters images by metadata or masks pixels by band value using the selected Mode and its parameters, and is commonly used for cloud filtering or threshold masking.\n\n"
        "**Filter by property:** " + knut.VALUE_FILTER_PROPERTY_FORMULA + "\n\n"
        "**Mask by band value:** "
        + knut.MASK_BY_BAND_FORMULA
        + " Or **bitwise0**: "
        + knut.MASK_BITWISE0_FORMULA
    )

    filter_mode = knext.StringParameter(
        "Mode",
        "Filter by property = drop entire images; Mask by band value = updateMask per image.",
        default_value="Filter by property",
        enum=["Filter by property", "Mask by band value"],
    )

    property_name = knext.StringParameter(
        "Property name",
        "Image metadata property. " + knut.VALUE_FILTER_PROPERTY_EXAMPLE,
        default_value="CLOUD_COVER",
    ).rule(knext.OneOf(filter_mode, ["Filter by property"]), knext.Effect.SHOW)

    property_operator = knext.StringParameter(
        "Property operator",
        "Comparison for the property value.",
        default_value="Less or Equal",
        enum=[
            "Equals",
            "Not Equals",
            "Greater Than",
            "Less Than",
            "Greater or Equal",
            "Less or Equal",
        ],
    ).rule(knext.OneOf(filter_mode, ["Filter by property"]), knext.Effect.SHOW)

    property_value = knext.StringParameter(
        "Property value",
        "Value to compare against (numeric or string).",
        default_value="20",
    ).rule(knext.OneOf(filter_mode, ["Filter by property"]), knext.Effect.SHOW)

    band_name = knext.StringParameter(
        "Band name",
        "Band to use for the mask. " + knut.MASK_BY_BAND_EXAMPLE,
        default_value="QA_RADSAT",
    ).rule(knext.OneOf(filter_mode, ["Mask by band value"]), knext.Effect.SHOW)

    band_operator = knext.StringParameter(
        "Band operator",
        "Comparison: pixels satisfying this are kept. bitwise0: "
        + knut.MASK_BITWISE0_EXAMPLE,
        default_value="==",
        enum=[">=", ">", "<=", "<", "==", "!=", "bitwise0"],
    ).rule(knext.OneOf(filter_mode, ["Mask by band value"]), knext.Effect.SHOW)

    band_value = knext.DoubleParameter(
        "Value",
        "Value to compare the band against (for bitwise0: integer to AND with band). "
        + knut.MASK_BITWISE0_EXAMPLE,
        default_value=0.0,
    ).rule(knext.OneOf(filter_mode, ["Mask by band value"]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        image_collection = ic_connection.image_collection

        if self.filter_mode == "Mask by band value":
            band = (self.band_name or "").strip()
            if not band:
                raise ValueError(
                    "Band name is required when Mode = Mask by band value."
                )
            val = self.band_value
            if self.band_operator == "bitwise0":
                and_val = int(val)
                if and_val < 1:
                    raise ValueError(
                        "Value for bitwise0 must be a positive integer (e.g. 31)."
                    )

                def mask_image(img):
                    mask = img.select(band).bitwiseAnd(and_val).eq(0)
                    return img.updateMask(mask)

                result = image_collection.map(mask_image)
                LOGGER.warning(
                    "Image Collection Value Filter (Mask by band): band=%s bitwise0 value=%s",
                    band,
                    and_val,
                )
            else:
                th = ee.Number(val)
                ops = {
                    ">=": lambda img: img.select(band).gte(th),
                    ">": lambda img: img.select(band).gt(th),
                    "<=": lambda img: img.select(band).lte(th),
                    "<": lambda img: img.select(band).lt(th),
                    "==": lambda img: img.select(band).eq(th),
                    "!=": lambda img: img.select(band).neq(th),
                }
                op_fn = ops[self.band_operator]

                def mask_image(img):
                    mask = op_fn(img)
                    return img.updateMask(mask)

                result = image_collection.map(mask_image)
                LOGGER.warning(
                    "Image Collection Value Filter (Mask by band): band=%s %s %s",
                    band,
                    self.band_operator,
                    val,
                )
            return knut.export_gee_image_collection_connection(result, ic_connection)

        # Filter by property
        prop_name = (self.property_name or "").strip()
        val_str = (self.property_value or "").strip()
        if not prop_name or val_str is None:
            LOGGER.warning(
                "Property name or value empty; returning original collection."
            )
            return knut.export_gee_image_collection_connection(
                image_collection, ic_connection
            )

        is_numeric = False
        numeric_value = None
        try:
            numeric_value = float(val_str)
            if (
                numeric_value != float("inf")
                and numeric_value != float("-inf")
                and numeric_value == numeric_value
            ):
                is_numeric = True
        except ValueError:
            pass

        if self.property_operator == "Equals":
            the_filter = (
                ee.Filter.eq(prop_name, numeric_value)
                if is_numeric
                else ee.Filter.eq(prop_name, val_str)
            )
        elif self.property_operator == "Not Equals":
            the_filter = (
                ee.Filter.neq(prop_name, numeric_value)
                if is_numeric
                else ee.Filter.neq(prop_name, val_str)
            )
        else:
            if not is_numeric:
                raise ValueError(
                    f"Operator '{self.property_operator}' requires a numeric value, got '{val_str}'."
                )
            operator_map = {
                "Greater Than": ee.Filter.gt,
                "Less Than": ee.Filter.lt,
                "Greater or Equal": ee.Filter.gte,
                "Less or Equal": ee.Filter.lte,
            }
            the_filter = operator_map[self.property_operator](prop_name, numeric_value)

        image_collection = image_collection.filter(the_filter)
        LOGGER.warning(
            "Image Collection Value Filter (by property): %s %s %s",
            prop_name,
            self.property_operator,
            val_str,
        )
        return knut.export_gee_image_collection_connection(
            image_collection, ic_connection
        )


############################################
# Image Collection Spatial Filter
############################################


@knext.node(
    name="GEE Image Collection Spatial Filter",
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
    """Filters and clips an Image Collection using a region of interest.

    This node provides spatial filtering and clipping capabilities for Image Collections.
    It can filter images that intersect with an ROI and optionally clip each image to the ROI boundary.
    This is essential for focusing analysis on specific geographic areas.

    **Spatial Operations:**

    - **Filter by Bounds**: Keep only images that intersect the ROI
    - **Clip to ROI**: Clip each image to the exact ROI boundary
    - **Combined**: Filter by bounds and clip simultaneously. Select both options to enable this.

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
        "Filter by bounds",
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
# Image Collection Band Calculator
############################################


def _simple_pattern_to_regex(pat):
    """Convert simple pattern (only . and .*) to a safe regex. . = one char, .* = any sequence."""
    import re

    placeholder = "\x00\x01"
    s = pat.replace(".*", placeholder)
    s = re.escape(s)
    s = s.replace(placeholder, ".*")
    s = s.replace(r"\.", ".")
    return s


@knext.node(
    name="GEE Image Collection Band Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICCalculator.png",
    id="imagecollectionbandcalculator",
    after="imagecollectionbandrenamer",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection (expression applied per image).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Image Collection with new band added from expression.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionBandCalculator:
    """Applies an expression to compute new bands and adds them to each image.

    This node computes a new band per image using Expression plus optional Band pattern/Output name parameters, and is commonly used to scale reflectance or compute indices.

    **Use simple regex pattern:** When enabled, use one band name pattern (e.g. ``SR_B.`` or ``ST_B.*``).
    Only ``.`` (one character) and ``.*`` (zero or more) are special. The expression must use **x** as
    the band variable (e.g. ``BX * 0.0000275 - 0.2``). All matching bands are updated in place (same name).

    Otherwise: expression uses band names and/or **prop(\"property_name\")**; one output band name.
    """

    use_simple_pattern = knext.BoolParameter(
        "Use simple regex pattern",
        "When checked, one band pattern (. and .* only); expression uses variable BX; matching bands updated in place.",
        default_value=False,
    )

    band_pattern = knext.StringParameter(
        "Band pattern",
        "Band name pattern: . = one character, .* = zero or more (e.g. SR_B. or ST_B.*). Only one pattern.",
        default_value="SR_B.",
    ).rule(knext.OneOf(use_simple_pattern, [True]), knext.Effect.SHOW)

    expression = knext.StringParameter(
        "Expression",
        'Earth Engine expression. In simple regex pattern mode use variable **BX** for the band (e.g. BX * 0.0000275 - 0.2). Otherwise band names and/or prop("property_name").',
        default_value="BX * 10000",
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name of the band to add (overwrites if present). Ignored in simple regex pattern mode.",
        default_value="SR_B1_scaled",
    ).rule(knext.OneOf(use_simple_pattern, [False]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
    ):
        import re
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        expression = (self.expression or "").strip()
        output_band_name = (self.output_band_name or "calculated").strip()
        if not expression:
            raise ValueError("Expression is required.")

        if self.use_simple_pattern:
            pattern = (self.band_pattern or "").strip()
            if not pattern:
                raise ValueError(
                    "Band pattern is required when Use simple regex pattern is checked."
                )
            first_img = ic.first()
            all_band_names = first_img.bandNames().getInfo()
            if not all_band_names:
                raise ValueError("Image collection has no bands.")
            regex = _simple_pattern_to_regex(pattern)
            matching = [b for b in all_band_names if re.fullmatch(regex, b)]
            if not matching:
                raise ValueError(
                    "No band names matched pattern %r. Available: %s"
                    % (pattern, all_band_names)
                )

            # Expression uses BX as the band variable (one band only)
            def map_fn_pattern(img):
                current = img
                for b in matching:
                    band_dict = {"BX": current.select(b)}
                    result_band = current.expression(expression, band_dict)
                    current = current.addBands(result_band.rename(b), overwrite=True)
                return current

            result = ic.map(map_fn_pattern)
            LOGGER.warning(
                "Image Collection Band Calculator (simple regex pattern): pattern=%s, %s bands updated",
                pattern,
                len(matching),
            )
            return knut.export_gee_image_collection_connection(result, ic_connection)

        prop_pattern = re.compile(r'prop\s*\(\s*["\']([^"\']+)["\']\s*\)')
        prop_to_placeholder = {}
        placeholders_ordered = []

        def repl(m):
            pname = m.group(1)
            if pname not in prop_to_placeholder:
                prop_to_placeholder[pname] = f"_p{len(placeholders_ordered)}"
                placeholders_ordered.append(pname)
            return prop_to_placeholder[pname]

        modified_expr = prop_pattern.sub(repl, expression)
        python_keywords = {
            "and",
            "or",
            "not",
            "if",
            "else",
            "True",
            "False",
            "None",
            "abs",
            "sqrt",
            "pow",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "min",
            "max",
            "round",
            "floor",
            "ceil",
            "int",
            "float",
        }
        potential = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", modified_expr)
        band_names = [
            n
            for n in potential
            if n not in python_keywords and not re.match(r"_p\d+$", n)
        ]
        band_names = list(dict.fromkeys(band_names))

        def map_fn(img):
            band_dict = {}
            for b in band_names:
                band_dict[b] = img.select(b)
            for prop_name, placeholder in prop_to_placeholder.items():
                band_dict[placeholder] = ee.Image.constant(img.get(prop_name))
            result_band = img.expression(modified_expr, band_dict)
            return img.addBands(result_band.rename(output_band_name), overwrite=True)

        result = ic.map(map_fn)
        LOGGER.warning(
            "Image Collection Band Calculator: output band '%s'",
            output_band_name,
        )
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection MultiBand Calculator
############################################


@knext.node(
    name="GEE Image Collection MultiBand Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICMultiCalculator.png",
    id="imagecollectionmultibandcalculator",
    after="imagecollectionbandcalculator",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection (expression applied per row per image).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection with new bands added from batch expression.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionMultiBandCalculator:
    """Apply a template expression to multiple bands.

    This node batch-applies a template expression to multiple bands using Expression template and Rows parameters, and is commonly used to scale or transform many bands at once.

    Use **BX1, BX2, BX3, BX4** as placeholders. **Rows are separated by semicolon (;)**;
    within each row, values are comma-separated (one row per output band).

    **Expression:** BX1 is the band placeholder (write **BX1** without quotes).
    For property names use **prop(\"BX2\")**, **prop(\"BX3\")** (keep the quotes around BX2/BX3).

    Examples:
    - 1 slot: expression ``BX1 * 0.0000275 - 0.2``, rows ``SR_B1; SR_B2; SR_B3``
    - 3 slots: expression ``BX1 * prop(\"BX2\") + prop(\"BX3\")``, rows
      ``SR_B1, REFLECTANCE_MULT_BAND_1, REFLECTANCE_ADD_BAND_1; SR_B2, REFLECTANCE_MULT_BAND_2, REFLECTANCE_ADD_BAND_2``
    """

    expression = knext.StringParameter(
        "Expression template",
        'BX1 = band (no quotes); property placeholders in prop("BX2"), prop("BX3"). E.g. BX1 * prop("BX2") + prop("BX3") or BX1 * 0.0000275 - 0.2.',
        default_value="BX1 + BX2",
    )

    rows = knext.StringParameter(
        "Rows (one per output band)",
        "Rows separated by semicolon (;); within each row, comma-separated values for BX1, BX2, ... (e.g. B1, B1_scaled;B2, B2_scaled).",
        default_value="B1, B1_scaled;B2, B2_scaled; ",
    )

    output_band_names = knext.StringParameter(
        "Output band names",
        "Comma-separated names for each row's output band. If empty, use first column (BX1) of each row.",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
    ):
        import re
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        expression = (self.expression or "").strip()
        rows_str = (self.rows or "").strip()
        if not expression:
            raise ValueError("Expression template is required.")
        if not rows_str:
            raise ValueError("Rows are required.")

        # Infer number of columns from BX1, BX2, ... in expression
        bx_matches = list(re.finditer(r"BX(\d+)", expression, re.IGNORECASE))
        if not bx_matches:
            raise ValueError(
                "Expression must contain at least one placeholder BX1, BX2, ..."
            )
        n_cols = max(int(m.group(1)) for m in bx_matches)

        # Parse rows (semicolon as row separator only)
        lines = [ln.strip() for ln in rows_str.split(";") if ln.strip()]
        row_list = []
        for ln in lines:
            vals = [v.strip() for v in ln.split(",") if v.strip()]
            if len(vals) < n_cols:
                raise ValueError(
                    f"Row has {len(vals)} values but expression needs {n_cols} (BX1..BX{n_cols})."
                )
            row_list.append(vals[:n_cols])

        # Output band names
        out_names_str = (self.output_band_names or "").strip()
        if out_names_str:
            output_band_names = [
                n.strip() for n in out_names_str.split(",") if n.strip()
            ]
            if len(output_band_names) != len(row_list):
                raise ValueError("Output band names count must match number of rows.")
        else:
            output_band_names = [row[0] for row in row_list]

        prop_pattern = re.compile(r'prop\s*\(\s*["\']([^"\']+)["\']\s*\)')
        python_keywords = {
            "and",
            "or",
            "not",
            "if",
            "else",
            "True",
            "False",
            "None",
            "abs",
            "sqrt",
            "pow",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "min",
            "max",
            "round",
            "floor",
            "ceil",
            "int",
            "float",
        }

        def build_modified_expr_and_dict(substituted_expr, img):
            """Replace prop('...') with _p0, _p1; return modified_expr and band_dict."""
            prop_to_placeholder = {}
            placeholders_ordered = []

            def repl(m):
                pname = m.group(1)
                if pname not in prop_to_placeholder:
                    prop_to_placeholder[pname] = f"_p{len(placeholders_ordered)}"
                    placeholders_ordered.append(pname)
                return prop_to_placeholder[pname]

            modified_expr = prop_pattern.sub(repl, substituted_expr)
            potential = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", modified_expr)
            band_names = [
                n
                for n in potential
                if n not in python_keywords and not re.match(r"_p\d+$", n)
            ]
            band_names = list(dict.fromkeys(band_names))
            band_dict = {}
            for b in band_names:
                band_dict[b] = img.select(b)
            for prop_name, placeholder in prop_to_placeholder.items():
                band_dict[placeholder] = ee.Image.constant(img.get(prop_name))
            return modified_expr, band_dict

        # Replace BXi from high to low so BX1 does not match inside BX10
        replace_order = sorted(range(1, n_cols + 1), reverse=True)

        def map_fn(img):
            current = img
            for row, out_name in zip(row_list, output_band_names):
                substituted = expression
                for i in replace_order:
                    substituted = substituted.replace(f"BX{i}", row[i - 1])
                modified_expr, band_dict = build_modified_expr_and_dict(
                    substituted, current
                )
                result_band = current.expression(modified_expr, band_dict)
                current = current.addBands(result_band.rename(out_name), overwrite=True)
            return current

        result = ic.map(map_fn)
        LOGGER.warning(
            "Image Collection MultiBand Calculator: %s bands",
            len(row_list),
        )
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection Band Selector
############################################


@knext.node(
    name="GEE Image Collection Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandSelector.png",
    id="imagecollectionbandselector",
    after="imagecollectionvaluefilter",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection to select bands from.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection with selected bands only.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionBandSelector:
    """Select a subset of bands for each image in the collection without renaming.

    This node selects a subset of bands using Bands and Use Regex parameters, and is commonly used to reduce data size before downstream processing.

    **Bands:** Comma-separated band names, or a single regex when **Use Regex** is checked.
    Regex uses Java syntax (matched on the GEE server), e.g. ``SR_B.`` or ``SR_B[0-9]``.
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated band names (e.g. SR_B1, SR_B2), or a single regex (e.g. SR_B.) when **Use Regex** is checked.",
        default_value="SR_B1, SR_B2",
    )

    use_regex = knext.BoolParameter(
        "Use Regex",
        "Interpret Bands as a Java regex pattern (matched on the GEE server). Uncheck for comma-separated names.",
        default_value=False,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        bands_str = (self.bands or "").strip()

        if self.use_regex:
            if not bands_str:
                raise ValueError("Bands is required when Use Regex is checked.")
            regex = bands_str

            def map_fn(img):
                return img.select(regex)

        else:
            if not bands_str:
                raise ValueError("Bands are required when Use Regex is unchecked.")
            band_list = [b.strip() for b in bands_str.split(",") if b.strip()]
            ee_band_list = ee.List(band_list)

            def map_fn(img):
                return img.select(ee_band_list)

        result = ic.map(map_fn)
        LOGGER.warning(
            "Image Collection Band Selector: use_regex=%s",
            self.use_regex,
        )
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection Band Renamer
############################################


@knext.node(
    name="GEE Image Collection Band Renamer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICRenamer.png",
    id="imagecollectionbandrenamer",
    after="imagecollectionbandselector",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection to select/rename bands.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection with selected/renamed bands.",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionBandRenamer:
    """Select bands by name and optionally rename them for each image in the collection.

    This node selects and optionally renames bands using Band names and New names parameters, and is commonly used to standardize band naming across sensors.

    Comma-separated band names to keep; optionally provide new names (same count) to rename.
    """

    band_names = knext.StringParameter(
        "Band names",
        "Comma-separated band names to keep (e.g. SR_B2, SR_B3).",
        default_value="SR_B2, SR_B3, SR_B4",
    )

    new_names = knext.StringParameter(
        "New names",
        "Comma-separated new names (same count as selected bands). Leave empty to keep original names.",
        default_value="Blue, Green, Red",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        bands_str = (self.band_names or "").strip()
        if not bands_str:
            raise ValueError("Band names are required.")
        band_list = [b.strip() for b in bands_str.split(",") if b.strip()]
        new_str = (self.new_names or "").strip()
        new_list = (
            [n.strip() for n in new_str.split(",") if n.strip()] if new_str else []
        )
        if new_list and len(new_list) != len(band_list):
            raise ValueError(
                "New names count must match band names count when renaming."
            )
        ee_band_list = ee.List(band_list)
        if new_list:
            ee_new_list = ee.List(new_list)

            def map_fn(img):
                return img.select(ee_band_list).rename(ee_new_list)

        else:

            def map_fn(img):
                return img.select(ee_band_list)

        result = ic.map(map_fn)
        LOGGER.warning("Image Collection Band Renamer applied.")
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection Merger
############################################


@knext.node(
    name="GEE Image Collection Merger",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICMerge.png",
    id="imagecollectionmerger",
    after="imagecollectionmultibandcalculator",
)
@knext.input_port(
    name="GEE Image Collection (primary)",
    description="First Image Collection (e.g. Landsat 8).",
    port_type=gee_image_collection_port_type,
)
@knext.input_port(
    name="GEE Image Collection (secondary)",
    description="Second Image Collection to merge (e.g. Landsat 7 with bands renamed to match).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Merged Image Collection (primary + secondary).",
    port_type=gee_image_collection_port_type,
)
class ImageCollectionMerger:
    """Merges two Image Collections into one.

    This node merges two Image Collections with no parameters and is commonly used to combine multi-sensor collections after band standardization.

    Uses Earth Engine's ``merge()``: the result contains all images from both
    collections. Use this to combine multi-sensor data (e.g. Landsat 8 + Landsat 7)
    before aggregating to a single composite.

    **Requirements:**

    - Both collections must have the **same band names and count** (use **Image
      Collection Band Selector** and **Image Collection Band Renamer** upstream so
      the secondary collection matches the primary).

    **Typical workflow:**

    1. Build L8 pipeline: filter → cloud mask → scaling → Band Selector (e.g. 6 bands).
    2. Build L7 pipeline: filter → cloud mask → scaling → Band Renamer (L7 names → L8 names).
    3. Connect both to this node (L8 primary, L7 secondary).
    4. Downstream: Image Collection Aggregator (median), then optional clip.
    """

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection_primary,
        ic_connection_secondary,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic1 = ic_connection_primary.image_collection
        ic2 = ic_connection_secondary.image_collection
        merged = ic1.merge(ic2)
        LOGGER.warning(
            "Image Collection Merger: merged two collections (order: primary, then secondary)."
        )
        return knut.export_gee_image_collection_connection(
            merged, ic_connection_primary
        )


############################################
# Image Collection Aggregator
############################################


@knext.node(
    name="GEE Image Collection Aggregator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageCollectionAggregator.png",
    id="imagecollectionaggregator",
    after="imagecollectionmerger",
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

    This node aggregates an Image Collection into a single composite using Aggregation method and Percentile(s) parameters, and is commonly used to build cloud-free or seasonal composites.

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
    - **count**: Per-pixel count of valid observations (useful for data availability maps)
    - **percentile**: Per-pixel percentile(s), e.g. 30th for cloud-free composites (see Percentile(s) parameter)

    **Common Use Cases:**

    - Create cloud-free composites using median
    - Calculate temporal averages for change detection
    - Find maximum NDVI values over a growing season
    - Generate annual precipitation sums
    - Create mosaics from overlapping images

    **Best Practices:**

    - Use **median** or **percentile** (e.g. 30) for cloud removal in optical imagery
    - Use **mean** for temporal averaging
    - Use **mosaic** for seamless image mosaicking
    - Apply filters before aggregation to improve results
    """

    aggregation_method = knext.StringParameter(
        "Aggregation method",
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
        - **count**: Per-pixel count of valid observations (data availability map)
        - **percentile**: Per-pixel percentile(s); use the Percentile(s) parameter to set value(s), e.g. 30 or 0,10,20,...,80
        """,
        default_value="first",
        enum=[
            "first",
            "last",
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "mode",
            "mosaic",
            "count",
            "percentile",
        ],
    )

    percentile_values = knext.StringParameter(
        "Percentile(s)",
        "Comma-separated percentile(s) in 0–100, e.g. 30 or 0,10,20,30,40,50,60,70,80. "
        "Lower percentiles (e.g. 30) often give cleaner, less cloudy composites.",
        default_value="30",
    ).rule(knext.OneOf(aggregation_method, ["percentile"]), knext.Effect.SHOW)

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
            "count": lambda ic: ic.count(),
        }

        # Methods that use reduce() produce images with undefined nominalScale.
        # Reproject to first image's projection/scale so downstream nodes get correct scale.
        _REDUCE_METHODS = {
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "mode",
            "count",
            "percentile",
        }

        # Apply aggregation
        try:
            if self.aggregation_method == "percentile":
                parts = [
                    s.strip() for s in self.percentile_values.split(",") if s.strip()
                ]
                percentiles = [int(p) for p in parts]
                if not percentiles or any(p < 0 or p > 100 for p in percentiles):
                    raise ValueError(
                        "Percentile(s) must be comma-separated integers in 0–100, e.g. 30 or 0,10,20,30,40,50,60,70,80"
                    )
                reducer = ee.Reducer.percentile(percentiles)
                image = image_collection.reduce(reducer)
            else:
                image = aggregation_methods[self.aggregation_method](image_collection)

            if self.aggregation_method in _REDUCE_METHODS:
                first_img = image_collection.first()
                ref_proj = first_img.select(0).projection()
                scale_m = ref_proj.nominalScale()
                image = image.reproject(crs=ref_proj, scale=scale_m)
                LOGGER.warning(
                    f"Aggregated using '{self.aggregation_method}', reprojected to first image scale ({scale_m}m)"
                )
            else:
                LOGGER.warning(
                    f"Successfully aggregated using '{self.aggregation_method}' method"
                )
        except Exception as e:
            LOGGER.error(
                f"Aggregation method '{self.aggregation_method}' failed: {e}. Falling back to 'first'."
            )
            image = image_collection.first()

        return knut.export_gee_image_connection(image, ic_connection)


############################################
# Image Collection Time-Window Aggregator
############################################


@knext.node(
    name="GEE Time-Window Aggregator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "TimeWindow.png",
    id="imagecollectiontimewindowaggregator",
    after="imagecollectionaggregator",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection with one aggregated image per time window (day/week/month/year).",
    port_type=gee_image_collection_port_type,
)
class TimeWindowAggregator:
    """Aggregates an Image Collection by time windows such as day, week, month, or year into a new Image Collection.

    This node aggregates images into fixed time windows using Time window and Aggregation method parameters, and is commonly used for monthly or annual composites.

    Each time window in the input collection's date range produces one image by applying the chosen
    aggregation method (same options as **Image Collection Aggregator**). Use **Image Collection
    General Filter** (or other filter nodes) upstream to set the date range.

    **Time window:** Day, Week, Month, or Year.

    **Aggregation method:** Same as Image Collection Aggregator (mean, median, sum, first, last, min, max, etc.).
    """

    time_window = knext.StringParameter(
        "Time window",
        "Interval to aggregate over: one output image per day, week, month, or year.",
        default_value="month",
        enum=["day", "week", "month", "year"],
    )

    aggregation_method = knext.StringParameter(
        "Aggregation method",
        "Method to aggregate images within each time window (same as Image Collection Aggregator).",
        default_value="median",
        enum=[
            "first",
            "last",
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "mode",
            "mosaic",
            "count",
            "percentile",
        ],
    )

    percentile_values = knext.StringParameter(
        "Percentile(s)",
        "Comma-separated percentile(s) in 0–100, e.g. 30 or 0,10,20,30,40,50,60,70,80.",
        default_value="30",
    ).rule(knext.OneOf(aggregation_method, ["percentile"]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging
        from datetime import datetime, timedelta

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection

        t_min_ms = ic.aggregate_min("system:time_start").getInfo()
        t_max_ms = ic.aggregate_max("system:time_start").getInfo()
        if t_min_ms is None or t_max_ms is None:
            raise ValueError(
                "Image collection has no images or no system:time_start. "
                "Use Image Collection General Filter (or similar) to provide a filtered collection with a date range."
            )
        start_dt = datetime.utcfromtimestamp(t_min_ms / 1000.0)
        end_dt = datetime.utcfromtimestamp(t_max_ms / 1000.0)

        window = self.time_window
        period_starts = []
        if window == "year":
            for y in range(start_dt.year, end_dt.year + 1):
                period_starts.append(datetime(y, 1, 1))
        elif window == "month":
            y, m = start_dt.year, start_dt.month
            while (y, m) <= (end_dt.year, end_dt.month):
                period_starts.append(datetime(y, m, 1))
                m += 1
                if m > 12:
                    m = 1
                    y += 1
        elif window == "week":
            d = start_dt.date()
            end_d = end_dt.date()
            while d <= end_d:
                period_starts.append(datetime.combine(d, datetime.min.time()))
                d += timedelta(days=7)
        else:  # day
            d = start_dt.date()
            end_d = end_dt.date()
            while d <= end_d:
                period_starts.append(datetime.combine(d, datetime.min.time()))
                d += timedelta(days=1)

        if not period_starts:
            raise ValueError(
                "No time windows in the input collection date range. "
                "Ensure the collection has a valid date range (e.g. use Image Collection General Filter)."
            )

        # Build end of each period for filterDate (exclusive end in GEE)
        def period_end(start, w):
            if w == "year":
                return start.replace(year=start.year + 1, month=1, day=1)
            if w == "month":
                if start.month == 12:
                    return start.replace(year=start.year + 1, month=1, day=1)
                return start.replace(month=start.month + 1, day=1)
            if w == "week":
                return start + timedelta(days=7)
            return start + timedelta(days=1)

        aggregation_methods = {
            "first": lambda fc: fc.first(),
            "last": lambda fc: fc.sort("system:time_start", False).first(),
            "mean": lambda fc: fc.mean(),
            "median": lambda fc: fc.median(),
            "min": lambda fc: fc.min(),
            "max": lambda fc: fc.max(),
            "sum": lambda fc: fc.sum(),
            "mode": lambda fc: fc.mode(),
            "mosaic": lambda fc: fc.mosaic(),
            "count": lambda fc: fc.count(),
        }

        use_percentile = self.aggregation_method == "percentile"
        if use_percentile:
            parts = [s.strip() for s in self.percentile_values.split(",") if s.strip()]
            percentiles = [int(p) for p in parts]
            if not percentiles or any(p < 0 or p > 100 for p in percentiles):
                raise ValueError(
                    "Percentile(s) must be comma-separated integers in 0–100."
                )
            reducer = ee.Reducer.percentile(percentiles)

        _REDUCE_METHODS = {
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "mode",
            "count",
            "percentile",
        }
        need_reproject = self.aggregation_method in _REDUCE_METHODS
        if need_reproject:
            first_img = ic.first()
            ref_proj = first_img.select(0).projection()
            scale_m = ref_proj.nominalScale()

        images = []
        for start_py in period_starts:
            end_py = period_end(start_py, window)
            start_ee = ee.Date(start_py.strftime("%Y-%m-%d"))
            end_ee = ee.Date(end_py.strftime("%Y-%m-%d"))
            filtered = ic.filterDate(start_ee, end_ee)
            if use_percentile:
                img = filtered.reduce(reducer)
            else:
                img = aggregation_methods[self.aggregation_method](filtered)
            if need_reproject:
                img = img.reproject(crs=ref_proj, scale=scale_m)
            start_ms = int(start_py.timestamp() * 1000)
            end_ms = int(end_py.timestamp() * 1000)
            img = (
                img.set("system:time_start", start_ms)
                .set("system:time_end", end_ms)
                .set("year", start_py.year)
            )
            if window == "month":
                img = img.set("month", start_py.month)
            images.append(img)

        result = ee.ImageCollection.fromImages(ee.List(images))
        LOGGER.warning(
            "Time-Window Aggregator: %s windows, method=%s",
            len(images),
            self.aggregation_method,
        )
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection Time Series Extractor
############################################


@knext.node(
    name="GEE Pixel-Band Time Series Extractor",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "TimeSeriesExtractor.png",
    id="imagecollectiontimeseriesextractor",
    after="imagecollectiontimeseriesregressor",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection with time series images.",
    port_type=gee_image_collection_port_type,
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection defining the region. The centroid of all features will be used as the extraction point.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_table(
    name="Time Series Table",
    description="Table containing time series data with date and band values.",
)
class ICTimeSeriesExtractor:
    """Extracts time series pixel values from an Image Collection at a point location.

    This node extracts pixel time series at a point using Band name(s) and a Feature Collection centroid, and is commonly used for temporal trend analysis and plotting.

    Extracts pixel values for one or more bands (comma-separated) from each image at the
    centroid of the input Feature Collection. Leave band name(s) empty to use all bands. This is useful for analyzing temporal trends,
    monitoring changes over time, and creating time series visualizations.

    **Input Requirements:**

    - **Image Collection**: Filtered Image Collection (date filtering can be done with filter nodes)
    - **Feature Collection**: Any geometry type (point, polygon, etc.) - the centroid will be used

    **Output:**

    - Table with columns: **date** and one column per band (e.g. **B6**, **B5**, **B4**)
    - Each row represents one time point from the Image Collection
    - Leave band name(s) empty to extract **all bands**

    **Use Cases:**

    - Monitor vegetation indices (NDVI, EVI) over time
    - Track land cover changes
    - Analyze seasonal patterns
    - Create time series charts for specific locations

    **Workflow Example:**

    1. Filter Image Collection: `Image Collection Reader` → `Image Collection General Filter` (date range)
    2. Select band: `Image Collection Band Selector` (optional, or specify in this node)
    3. Define region: `Feature Collection Reader` or create point/polygon
    4. Extract time series: This node
    5. Visualize: Use KNIME's plotting nodes to create time series charts
    """

    band = knext.StringParameter(
        "Band name(s)",
        "Comma-separated band names (e.g. 'B6,B5,B4' or 'NDVI,EVI'). Leave empty to use all bands in the collection.",
        default_value="",
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_connection,
        fc_connection,
    ):
        import ee
        import logging
        import pandas as pd
        from datetime import datetime

        LOGGER = logging.getLogger(__name__)

        try:
            # Get Image Collection and Feature Collection
            image_collection = ic_connection.image_collection
            feature_collection = fc_connection.feature_collection

            if not isinstance(image_collection, ee.ImageCollection):
                raise ValueError("Input must be an ImageCollection")

            if not isinstance(feature_collection, ee.FeatureCollection):
                raise ValueError("Input must be a FeatureCollection")

            # Get centroid of Feature Collection (unaryUnion then centroid)
            LOGGER.warning("Calculating centroid of Feature Collection...")
            centroid = feature_collection.geometry().centroid()
            centroid_coords = centroid.coordinates().getInfo()
            LOGGER.warning(f"Using centroid point: {centroid_coords}")

            # Create a point FeatureCollection for extraction
            point_fc = ee.FeatureCollection([ee.Feature(centroid, {"id": "point"})])

            # Get the first image to determine scale and resolve band list
            first_image = image_collection.first()
            band_names = first_image.bandNames().getInfo()
            LOGGER.warning("First image has %s bands: %s", len(band_names), band_names)

            bands_str = (self.band or "").strip()
            if not bands_str:
                bands_to_use = band_names
                LOGGER.warning("No bands specified; using all bands: %s", bands_to_use)
            else:
                bands_to_use = [b.strip() for b in bands_str.split(",") if b.strip()]
                missing = [b for b in bands_to_use if b not in band_names]
                if missing:
                    raise ValueError(
                        f"Band(s) {missing} not found in Image Collection. Available: {band_names}"
                    )

            # Get nominal scale from first image (use first selected band)
            try:
                scale = (
                    first_image.select(bands_to_use[0])
                    .projection()
                    .nominalScale()
                    .getInfo()
                )
                LOGGER.warning(f"Using automatic scale: {scale} meters")
            except Exception as e:
                LOGGER.warning(
                    f"Could not get nominal scale: {e}. Using default scale: 30 meters"
                )
                scale = 30

            # Select the specified band(s) from the collection
            band_collection = image_collection.select(bands_to_use)

            # Get collection size for progress tracking
            try:
                collection_size = image_collection.size().getInfo()
                LOGGER.warning(
                    f"Extracting time series from {collection_size} images at point {centroid_coords}"
                )
            except Exception:
                LOGGER.warning("Extracting time series (size check skipped)")

            # Use getRegion to extract time series efficiently
            LOGGER.warning(
                "Extracting time series for band(s) %s (scale=%s)...",
                bands_to_use,
                scale,
            )
            time_series_data = []
            try:
                region_data = band_collection.getRegion(
                    geometry=centroid,
                    scale=scale,
                    crs="EPSG:4326",
                ).getInfo()

                LOGGER.warning(
                    "getRegion returned %s rows (incl. header)",
                    len(region_data) if region_data else 0,
                )
                if not region_data or len(region_data) < 2:
                    raise ValueError(
                        "No data extracted from Image Collection (empty or single row)"
                    )

                headers = region_data[0]
                LOGGER.warning("getRegion headers: %s", headers)
                time_index = headers.index("time") if "time" in headers else None
                if time_index is None:
                    LOGGER.warning(
                        "No 'time' in getRegion headers; rows will be skipped"
                    )
                band_indices = {
                    b: headers.index(b) for b in bands_to_use if b in headers
                }
                if len(band_indices) != len(bands_to_use):
                    missing_in_resp = [b for b in bands_to_use if b not in band_indices]
                    raise ValueError(
                        f"Band(s) {missing_in_resp} not in getRegion output (headers: {headers})"
                    )

                rows_with_val = 0
                rows_skipped_no_date = 0
                rows_skipped_all_null = 0
                for row in region_data[1:]:
                    if (
                        time_index is None
                        or time_index >= len(row)
                        or row[time_index] is None
                    ):
                        date_str = None
                        rows_skipped_no_date += 1
                    else:
                        time_ms = row[time_index]
                        date_str = datetime.fromtimestamp(time_ms / 1000).strftime(
                            "%Y-%m-%d"
                        )
                    if not date_str:
                        continue
                    rec = {"date": date_str}
                    any_val = False
                    for b, idx in band_indices.items():
                        v = row[idx] if idx < len(row) else None
                        rec[b] = v
                        if v is not None:
                            any_val = True
                    if any_val:
                        time_series_data.append(rec)
                        rows_with_val += 1
                    else:
                        rows_skipped_all_null += 1

                LOGGER.warning(
                    "getRegion parse: %s rows with at least one value, %s skipped (all null), %s skipped (no date)",
                    rows_with_val,
                    rows_skipped_all_null,
                    rows_skipped_no_date,
                )

            except Exception as e:
                LOGGER.warning(
                    "getRegion method failed: %s. Falling back to iterative method...",
                    e,
                )
                time_series_data = []
                image_list = image_collection.toList(image_collection.size())
                image_list_size = image_list.size().getInfo()
                LOGGER.warning(
                    "Iterative fallback: processing %s images one by one",
                    image_list_size,
                )
                failed_count = 0
                for i in range(image_list_size):
                    try:
                        image = ee.Image(image_list.get(i))
                        time_start = image.get("system:time_start").getInfo()
                        date_str = datetime.fromtimestamp(time_start / 1000).strftime(
                            "%Y-%m-%d"
                        )
                        value_dict = (
                            image.select(bands_to_use)
                            .reduceRegion(
                                reducer=ee.Reducer.first(),
                                geometry=centroid,
                                scale=scale,
                                maxPixels=1e9,
                            )
                            .getInfo()
                        )
                        rec = {"date": date_str}
                        for b in bands_to_use:
                            rec[b] = value_dict.get(b)
                        if any(rec.get(b) is not None for b in bands_to_use):
                            time_series_data.append(rec)
                        if (i + 1) % 10 == 0:
                            LOGGER.warning(
                                "Processed %s/%s images, %s rows so far, %s failed",
                                i + 1,
                                image_list_size,
                                len(time_series_data),
                                failed_count,
                            )
                    except Exception as img_error:
                        failed_count += 1
                        LOGGER.warning(
                            "Failed to process image %s/%s: %s",
                            i + 1,
                            image_list_size,
                            img_error,
                        )
                        continue
                LOGGER.warning(
                    "Iterative fallback done: %s valid rows, %s images failed",
                    len(time_series_data),
                    failed_count,
                )

            if not time_series_data:
                LOGGER.warning(
                    "No valid time series data: centroid=%s, bands=%s, scale=%s, collection_size unknown (getRegion/iterative produced 0 rows)",
                    centroid_coords,
                    bands_to_use,
                    scale,
                )
                raise ValueError(
                    "No valid time series data extracted. Check that the point is within image bounds and the band(s) exist."
                )

            df = pd.DataFrame(time_series_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            LOGGER.warning(
                "Successfully extracted %s time series points for band(s) %s",
                len(df),
                bands_to_use,
            )

            return knext.Table.from_pandas(df)

        except Exception as e:
            LOGGER.error(f"Time series extraction failed: {e}")
            raise


############################################
# Add Time and Harmonic Bands – component options
############################################


class AddTimeHarmonicComponentOptions(knext.EnumParameterOptions):
    """Components that can be added to each image for time-series regression or analysis."""

    YEAR = (
        "Year",
        "Add a band with the image year (integer from system:time_start). Use with Band Calculator for e.g. t = year - 1970.",
    )
    DAY_OF_YEAR = (
        "Day of year",
        "Add a band with day of year 1–366 from system:time_start.",
    )
    MONTH = (
        "Month",
        "Add a band with month 1–12 from system:time_start.",
    )
    FRACTIONAL_YEARS = (
        "Fractional years",
        "Add a band with years since reference date (fractional), e.g. for harmonic t.",
    )
    CONSTANT = (
        "Constant",
        "Add a band with constant value (default 1) for regression intercept.",
    )
    HARMONIC_PAIRS = (
        "Harmonic pairs",
        "Add cos(2πkt), sin(2πkt) bands (k=1..N) with t = year - reference_year.",
    )
    PROPERTY_AS_BAND = (
        "Image properties as bands",
        "Add bands from image metadata (e.g. CLOUD_COVER, SUN_ELEVATION). Format: PROPERTY:band_name, ...",
    )

    @classmethod
    def get_default(cls):
        return [cls.CONSTANT.name]


############################################
# Image Collection Add Time and Harmonic Bands
############################################


@knext.node(
    name="GEE Add Time and Harmonic Bands",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "AddBands.png",
    id="imagecollectionaddtimeharmonicbands",
    after="imagecollectiontimewindowaggregator",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection (e.g. Landsat with NDVI band).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Image Collection with added bands (time-derived, constant, harmonic, or image properties).",
    port_type=gee_image_collection_port_type,
)
class ICAddTimeHarmonicBands:
    """Adds time and harmonic bands to each image for time-series regression or generic analysis.

    This node adds time-derived and harmonic bands using Components/Reference year/K parameters, and is commonly used to prepare inputs for time-series regression.

    Choose one or more components via **Components to add**:

    - **Year**: Image year (integer). Use downstream for e.g. t = year - 1970.
    - **Day of year**: 1–366.
    - **Month**: 1–12.
    - **Fractional years**: Years since reference date (e.g. 1970-01-01), fractional.
    - **Constant**: Configurable value (e.g. 1 for intercept).
    - **Harmonic pairs**: cos(2πkt), sin(2πkt) with t = year - reference_year.
    - **Image properties as bands**: Add any image property (e.g. CLOUD_COVER, SUN_ELEVATION) as a constant band per image. Format: PROPERTY_NAME:output_band_name, comma-separated.
    """

    components_to_add = knext.EnumSetParameter(
        "Components to add",
        "Select one or more: Year, Day of year, Month, Fractional years, Constant, Harmonic pairs, or Image properties as bands.",
        default_value=AddTimeHarmonicComponentOptions.get_default(),
        enum=AddTimeHarmonicComponentOptions,
    )

    # Year – show only when YEAR is selected (use option class member for OneOf with EnumSet)
    year_band_name = knext.StringParameter(
        "Year band name",
        "Name of the band storing the image year (integer).",
        default_value="year",
    ).rule(
        knext.Contains(components_to_add, AddTimeHarmonicComponentOptions.YEAR.name),
        knext.Effect.SHOW,
    )

    # Day of year – show only when DAY_OF_YEAR is selected
    day_of_year_band_name = knext.StringParameter(
        "Day of year band name",
        "Name of the day-of-year band (1–366).",
        default_value="day_of_year",
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.DAY_OF_YEAR.name
        ),
        knext.Effect.SHOW,
    )

    # Month – show only when MONTH is selected
    month_band_name = knext.StringParameter(
        "Month band name",
        "Name of the month band (1–12).",
        default_value="month",
    ).rule(
        knext.Contains(components_to_add, AddTimeHarmonicComponentOptions.MONTH.name),
        knext.Effect.SHOW,
    )

    # Fractional years – show only when FRACTIONAL_YEARS is selected
    fractional_years_band_name = knext.StringParameter(
        "Fractional years band name",
        "Name of the band (years since reference date, fractional).",
        default_value="t",
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.FRACTIONAL_YEARS.name
        ),
        knext.Effect.SHOW,
    )

    fractional_years_reference_date = knext.StringParameter(
        "Reference date (fractional years)",
        "ISO date for t = 0 (e.g. 1970-01-01).",
        default_value="1970-01-01",
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.FRACTIONAL_YEARS.name
        ),
        knext.Effect.SHOW,
    )

    # Constant – show only when CONSTANT is selected
    constant_band_name = knext.StringParameter(
        "Constant band name",
        "Name of the constant band.",
        default_value="constant",
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.CONSTANT.name
        ),
        knext.Effect.SHOW,
    )

    constant_value = knext.DoubleParameter(
        "Constant value",
        "Value for the constant band (e.g. 1 for regression intercept).",
        default_value=1.0,
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.CONSTANT.name
        ),
        knext.Effect.SHOW,
    )

    # Harmonic – show only when HARMONIC_PAIRS is selected
    num_harmonics = knext.IntParameter(
        "Number of harmonic pairs",
        "Number of cos/sin pairs (1 = one cycle per year; 2+ = higher-order harmonics).",
        default_value=1,
        min_value=1,
        max_value=5,
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.HARMONIC_PAIRS.name
        ),
        knext.Effect.SHOW,
    )

    harmonic_reference_year = knext.IntParameter(
        "Reference year (for harmonic t)",
        "Harmonic uses t = image_year - reference_year (e.g. 1970).",
        default_value=1970,
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.HARMONIC_PAIRS.name
        ),
        knext.Effect.SHOW,
    )

    # Image properties as bands – show only when PROPERTY_AS_BAND is selected
    property_as_band_spec = knext.StringParameter(
        "Property → band (property:band_name, ...)",
        "Image property names and output band names, comma-separated (e.g. CLOUD_COVER:cloud_cover, SUN_ELEVATION:sun_elev). Missing properties may cause runtime errors.",
        default_value="",
    ).rule(
        knext.Contains(
            components_to_add, AddTimeHarmonicComponentOptions.PROPERTY_AS_BAND.name
        ),
        knext.Effect.SHOW,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging
        import math

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        selected = [getattr(s, "name", str(s)).upper() for s in self.components_to_add]
        add_year = "YEAR" in selected
        add_day_of_year = "DAY_OF_YEAR" in selected
        add_month = "MONTH" in selected
        add_fractional_years = "FRACTIONAL_YEARS" in selected
        add_constant = "CONSTANT" in selected
        add_harmonic = "HARMONIC_PAIRS" in selected
        add_property_as_band = "PROPERTY_AS_BAND" in selected

        if not (
            add_year
            or add_day_of_year
            or add_month
            or add_fractional_years
            or add_constant
            or add_harmonic
            or add_property_as_band
        ):
            raise ValueError("Select at least one component in Components to add.")

        # Parse property→band pairs: "PROP1:band1, PROP2:band2"
        prop_band_pairs = []
        if add_property_as_band and (self.property_as_band_spec or "").strip():
            for part in (self.property_as_band_spec or "").split(","):
                part = part.strip()
                if ":" in part:
                    prop, band = part.split(":", 1)
                    prop, band = prop.strip(), band.strip()
                    if prop and band:
                        prop_band_pairs.append((prop, band))
            if add_property_as_band and not prop_band_pairs:
                raise ValueError(
                    "Image properties as bands: specify at least one pair as PROPERTY_NAME:band_name (e.g. CLOUD_COVER:cloud_cover)."
                )

        year_name = (self.year_band_name or "year").strip()
        doy_name = (self.day_of_year_band_name or "day_of_year").strip()
        month_name = (self.month_band_name or "month").strip()
        frac_name = (self.fractional_years_band_name or "t").strip()
        ref_date_str = (self.fractional_years_reference_date or "1970-01-01").strip()
        const_name = (self.constant_band_name or "constant").strip()
        ref_year = self.harmonic_reference_year
        K = self.num_harmonics if add_harmonic else 0
        two_pi = 2 * math.pi
        ref_date_ee = ee.Date(ref_date_str)

        def add_bands(img):
            geom = img.geometry()
            # Match new bands to original image scale (constant images default to EPSG:4326 coarse)
            ref_proj = img.select(0).projection()
            scale_ref = ref_proj.nominalScale()
            date_ee = ee.Date(img.get("system:time_start"))
            year_ee = ee.Number(date_ee.get("year"))
            time_num = year_ee.subtract(ref_year)
            result = img

            def _same_scale(im):
                return im.clip(geom).reproject(crs=ref_proj, scale=scale_ref)

            # Use .toFloat() so band type is generic Float; otherwise EE treats constant bands as inhomogeneous (Float<v,v> per image) and getRegion/select fail.
            if add_year:
                result = result.addBands(
                    _same_scale(ee.Image.constant(year_ee).toFloat().rename(year_name))
                )
            if add_day_of_year:
                doy = ee.Number(date_ee.getRelative("day", "year")).add(1)
                result = result.addBands(
                    _same_scale(ee.Image.constant(doy).toFloat().rename(doy_name))
                )
            if add_month:
                month = ee.Number(date_ee.get("month"))
                result = result.addBands(
                    _same_scale(ee.Image.constant(month).toFloat().rename(month_name))
                )
            if add_fractional_years:
                frac_years = date_ee.difference(ref_date_ee, "year")
                result = result.addBands(
                    _same_scale(
                        ee.Image.constant(frac_years).toFloat().rename(frac_name)
                    )
                )
            if add_constant:
                result = result.addBands(
                    _same_scale(
                        ee.Image.constant(self.constant_value)
                        .toFloat()
                        .rename(const_name)
                    )
                )
            if add_harmonic:
                for k in range(1, K + 1):
                    angle = time_num.multiply(two_pi * k)
                    result = result.addBands(
                        _same_scale(
                            ee.Image.constant(angle.cos())
                            .toFloat()
                            .rename("cos" + str(k))
                        )
                    ).addBands(
                        _same_scale(
                            ee.Image.constant(angle.sin())
                            .toFloat()
                            .rename("sin" + str(k))
                        )
                    )
            for prop_name, band_name in prop_band_pairs:
                val = img.get(prop_name)
                band_img = ee.Image.constant(ee.Number(val)).toFloat().rename(band_name)
                result = result.addBands(_same_scale(band_img))

            return result

        result = ic.map(add_bands)
        parts = []
        if add_year:
            parts.append("year (%s)" % year_name)
        if add_day_of_year:
            parts.append("day_of_year (%s)" % doy_name)
        if add_month:
            parts.append("month (%s)" % month_name)
        if add_fractional_years:
            parts.append("fractional_years (%s)" % frac_name)
        if add_constant:
            parts.append("constant (%s)" % const_name)
        if add_harmonic:
            parts.append("%s harmonic pair(s)" % K)
        if add_property_as_band and prop_band_pairs:
            parts.append("properties (%s)" % ", ".join(b for _, b in prop_band_pairs))
        LOGGER.warning("Added: %s", ", ".join(parts))
        return knut.export_gee_image_collection_connection(result, ic_connection)


############################################
# Image Collection Time Series Regressor
############################################


@knext.node(
    name="GEE Image Collection Time Series Regressor",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICRegression.png",
    id="imagecollectiontimeseriesregressor",
    after="imagecollectionaddtimeharmonicbands",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="Image Collection with independent and dependent bands (e.g. from Add Time and Harmonic Bands + NDVI).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Image Collection with coefficient bands (regress) or + fitted/detrended (detrend mode).",
    port_type=gee_image_collection_port_type,
)
class ICTimeSeriesRegressor:
    """Runs per-pixel time regression and adds coefficients and fitted outputs.

    This node runs per-pixel time regression using Mode and independent/dependent band parameters, and is commonly used for trend estimation or detrending.

    **Mode regress:** Adds intercept, <indep>_coef, and fitted (trend) to each image.
    **Mode detrend:** Same as regress, plus detrended band (dependent − fitted). No extra option; both are always added.
    Single map over the collection in both modes.
    """

    mode = knext.StringParameter(
        "Mode",
        "Regress: add intercept, *_coef, and fitted. Detrend: also add detrended band.",
        default_value="detrend",
        enum=["regress", "detrend"],
    )

    independent_band_names = knext.StringParameter(
        "Independent band names",
        "Comma-separated band names in order (e.g. 'constant,t' or 'constant,t,cos1,sin1').",
        default_value="constant,t",
    )

    dependent_band_name = knext.StringParameter(
        "Dependent band name",
        "Single band to fit (e.g. 'NDVI', 'ndvi').",
        default_value="NDVI",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        indep_str = (self.independent_band_names or "").strip()
        dep_band = (self.dependent_band_name or "").strip()
        if not indep_str or not dep_band:
            raise ValueError("Independent and dependent band names are required.")
        indep_list = [b.strip() for b in indep_str.split(",") if b.strip()]
        num_x = len(indep_list)
        num_y = 1
        # GEE linearRegression returns coefficients of shape (numX, numY), not (numX+1, numY); intercept is the coef for the first band (e.g. constant)
        names = [b + "_coef" for b in indep_list]
        band_order = indep_list + [dep_band]
        ic_selected = ic.select(band_order)
        reducer = ee.Reducer.linearRegression(numX=num_x, numY=num_y)
        array_image = ic_selected.reduce(reducer)
        coeff_array = array_image.select("coefficients")
        sliced = coeff_array.arraySlice(0, 0, num_x)
        coef_image = sliced.arrayProject([0]).arrayFlatten([names]).toFloat()
        # Reproject coefficient image to match collection; use nominalScale so reduced image keeps input resolution
        first_img = ic.first()
        first_proj = first_img.select(band_order[0]).projection()
        scale_m = first_proj.nominalScale()
        try:
            scale_info = scale_m.getInfo()
            crs_info = (
                first_proj.crs().getInfo() if hasattr(first_proj, "crs") else "N/A"
            )
        except Exception as e:
            scale_info = "(getInfo failed: %s)" % e
            crs_info = "N/A"
        LOGGER.warning(
            "Time series regressor: first image projection scale_m=%s, crs=%s (used for coef_image.reproject)",
            scale_info,
            crs_info,
        )
        coef_image = coef_image.reproject(crs=first_proj, scale=scale_m)
        is_detrend = self.mode == "detrend"

        def add_bands(img):
            img_proj = img.select(band_order[0]).projection()
            coef_for_img = coef_image.reproject(
                crs=img_proj, scale=img_proj.nominalScale()
            )
            result = img.addBands(coef_for_img)
            # fitted = sum(independent * coefficient); first coef is intercept when first band is constant=1
            fitted = (
                result.select(indep_list)
                .multiply(result.select(names))
                .reduce(ee.Reducer.sum())
                .rename("fitted")
            )
            result = result.addBands(fitted)
            if is_detrend:
                detrended = result.select(dep_band).subtract(fitted).rename("detrended")
                result = result.addBands(detrended)
            # Force single projection so downstream (e.g. Pixel-Band Time Series Extractor) reduceRegion works
            return result.reproject(crs=img_proj, scale=img_proj.nominalScale())

        result_ic = ic.map(add_bands)
        bands_added = names + ["fitted"] + (["detrended"] if is_detrend else [])
        LOGGER.warning(
            "Time series regression: %s independent, 1 dependent → mode=%s, bands added: %s; band_order=%s",
            num_x,
            self.mode,
            bands_added,
            band_order,
        )
        LOGGER.warning(
            "Time series regressor: each image gets coef reprojected to img_proj+nominalScale, then result.reproject(img_proj, nominalScale) applied.",
        )
        return knut.export_gee_image_collection_connection(result_ic, ic_connection)


############################################
# Image Collection Lagged Join (self-lag)
############################################


@knext.node(
    name="GEE Image Collection Lagged Self-Join",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "SelfLag.png",
    id="imagecollectionlaggedjoin",
    after="imagecollectionaggregator",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="Image Collection (e.g. detrended Landsat with NDVI).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Image Collection with lagged bands (band_1, band_2, ...) merged into each image.",
    port_type=gee_image_collection_port_type,
)
class ICLaggedJoin:
    """Joins each image with earlier images and merges their bands.

    This node stacks bands from earlier images as lagged features using Band/Lag/Number of lags parameters, and is commonly used for autocorrelation or time-series modeling.

    Uses ee.Join.saveAll to find previous images within lag_days, then merges the first num_lags
    as band_1, band_2, ... For Landsat 16-day use lag_days=17 to get one previous image.
    """

    band_name = knext.StringParameter(
        "Band to merge from lagged images",
        "Band name to add as lagged (e.g. 'NDVI' → NDVI_1, NDVI_2).",
        default_value="NDVI",
    )

    lag_days = knext.IntParameter(
        "Lag (days)",
        "Time window in days (e.g. 17 for Landsat: one previous image).",
        default_value=17,
        min_value=1,
        max_value=365,
    )

    num_lags = knext.IntParameter(
        "Number of lagged images to merge",
        "How many previous images to stack (1 → band_1, 2 → band_1 and band_2).",
        default_value=1,
        min_value=1,
        max_value=10,
    )

    min_previous = knext.IntParameter(
        "Minimum previous images required",
        "Drop images with fewer than this many previous neighbors (0 = keep all).",
        default_value=0,
        min_value=0,
        max_value=10,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        band = (self.band_name or "").strip()
        if not band:
            raise ValueError("Band name is required.")
        lag_ms = int(self.lag_days) * 24 * 3600 * 1000
        num_lags = int(self.num_lags)
        min_prev = int(self.min_previous)

        def to_feature(img):
            return ee.Feature(
                ee.Image(img).geometry().bounds(),
                {
                    "system:time_start": ee.Image(img).get("system:time_start"),
                    "image": img,
                },
            )

        fc = ic.map(to_feature)
        join_filter = ee.Filter.And(
            ee.Filter.maxDifference(
                difference=lag_ms,
                leftField="system:time_start",
                rightField="system:time_start",
            ),
            ee.Filter.greaterThan(
                leftField="system:time_start",
                rightField="system:time_start",
            ),
        )
        join = ee.Join.saveAll(
            matchesKey="images",
            ordering="system:time_start",
            ascending=False,
        )
        joined_fc = join.apply(fc, fc, join_filter)

        def feature_with_merged(feature):
            primary = ee.Image(feature.get("image"))
            prev_list = ee.List(feature.get("images")).slice(0, num_lags)

            def add_prev(prev_feature, acc):
                acc_f = ee.Feature(acc)
                acc_img = ee.Image(acc_f.get("image"))
                k = ee.Number(acc_f.get("k"))
                prev_img = ee.Image(ee.Feature(prev_feature).get("image"))
                lag_band_name = ee.String(band).cat("_").cat(k.format())
                new_img = acc_img.addBands(
                    prev_img.select([band]).rename(ee.List([lag_band_name]))
                )
                return ee.Feature(None, {"image": new_img, "k": k.add(1)})

            start = ee.Feature(None, {"image": primary, "k": 1})
            merged_f = ee.List(prev_list).iterate(add_prev, start)
            merged = ee.Image(ee.Feature(merged_f).get("image"))
            n_prev = ee.List(feature.get("images")).size()
            return ee.Feature(None, {"image": merged, "n_prev": n_prev})

        joined_with_merged = joined_fc.map(feature_with_merged)
        if min_prev > 0:
            joined_with_merged = joined_with_merged.filter(
                ee.Filter.gte("n_prev", min_prev)
            )
        result_ic = ee.ImageCollection(
            joined_with_merged.map(lambda f: ee.Image(ee.Feature(f).get("image")))
        )
        LOGGER.warning(
            "Lagged Join: lag_days=%s, num_lags=%s, band=%s",
            self.lag_days,
            num_lags,
            band,
        )
        return knut.export_gee_image_collection_connection(result_ic, ic_connection)


############################################
# Image Collection Cross Lagged Join )
############################################


@knext.node(
    name="GEE Image Collection Cross Lagged Join",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "CrossLag.png",
    id="imagecollectioncrosslaggedjoin",
    after="imagecollectionlaggedjoin",
)
@knext.input_port(
    name="GEE Image Collection (primary)",
    description="Primary Image Collection (e.g. Landsat NDVI).",
    port_type=gee_image_collection_port_type,
)
@knext.input_port(
    name="GEE Image Collection (secondary)",
    description="Secondary Image Collection (e.g. precipitation) to join by time.",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Collection Connection",
    description="Primary collection with one added band from secondary (lagged or aggregated).",
    port_type=gee_image_collection_port_type,
)
class ICCrossLaggedJoin:
    """Joins images by time lag and adds bands from the secondary collection.

    This node joins a primary and secondary collection by time lag using Lag/Mode/Band parameters, and is commonly used to relate drivers and responses.

    **Single:** one secondary image (nearest before primary) → one band (e.g. precip_1, 5-day lag).
    **Sum:** all secondary images in the window are summed → one band (e.g. precip_sum_30d, 30-day lag).
    """

    lag_days = knext.IntParameter(
        "Lag (days)",
        "Time window: secondary images with date in (primary_date - lag_days, primary_date].",
        default_value=5,
        min_value=1,
        max_value=365,
    )

    secondary_band_name = knext.StringParameter(
        "Secondary band name",
        "Band from the secondary collection to add (e.g. precipitation band name).",
        default_value="precipitation",
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name for the added band (e.g. 'precip_1' for single, 'precip_sum_30d' for sum).",
        default_value="precip_1",
    )

    mode = knext.StringParameter(
        "Mode",
        "Single: use one secondary image (nearest before primary). Sum: aggregate all secondary in window (sum).",
        default_value="single",
        enum=["single", "sum"],
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        ic_primary_connection,
        ic_secondary_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic_primary = ic_primary_connection.image_collection
        ic_secondary = ic_secondary_connection.image_collection
        lag_ms = int(self.lag_days) * 24 * 3600 * 1000
        sec_band = (self.secondary_band_name or "").strip()
        out_band = (self.output_band_name or "").strip()
        if not sec_band or not out_band:
            raise ValueError("Secondary band name and output band name are required.")
        is_single = self.mode == "single"

        def to_feature(img):
            return ee.Feature(
                ee.Image(img).geometry().bounds(),
                {
                    "system:time_start": ee.Image(img).get("system:time_start"),
                    "image": img,
                },
            )

        fc_primary = ic_primary.map(to_feature)
        fc_secondary = ic_secondary.map(to_feature)
        join_filter = ee.Filter.And(
            ee.Filter.maxDifference(
                difference=lag_ms,
                leftField="system:time_start",
                rightField="system:time_start",
            ),
            ee.Filter.greaterThan(
                leftField="system:time_start",
                rightField="system:time_start",
            ),
        )
        join = ee.Join.saveAll(
            matchesKey="images",
            ordering="system:time_start",
            ascending=False,
        )
        joined_fc = join.apply(fc_primary, fc_secondary, join_filter)

        if is_single:

            def add_secondary_band_single(feature):
                primary_img = ee.Image(feature.get("image"))
                images = ee.List(feature.get("images"))
                n = images.size()
                no_data_band = ee.Image.constant(0).toFloat().rename([out_band])
                band_img = ee.Algorithms.If(
                    n.eq(0),
                    no_data_band,
                    ee.Image(ee.Feature(images.get(0)).get("image"))
                    .select([sec_band])
                    .rename([out_band]),
                )
                return primary_img.addBands(ee.Image(band_img))

            result_ic = ee.ImageCollection(joined_fc.map(add_secondary_band_single))
        else:

            def add_secondary_band_sum(feature):
                primary_img = ee.Image(feature.get("image"))
                images = ee.List(feature.get("images"))
                n = images.size()

                def to_img(f):
                    return ee.Image(ee.Feature(f).get("image")).select([sec_band])

                img_list = images.map(to_img)
                col = ee.ImageCollection.fromImages(img_list)
                summed = col.sum().rename([out_band])
                no_data_band = ee.Image.constant(0).toFloat().rename([out_band])
                band_img = ee.Algorithms.If(n.eq(0), no_data_band, summed)
                return primary_img.addBands(ee.Image(band_img))

            result_ic = ee.ImageCollection(joined_fc.map(add_secondary_band_sum))
        LOGGER.warning(
            "Cross Lagged Join: lag_days=%s, mode=%s, secondary_band=%s -> %s",
            self.lag_days,
            self.mode,
            sec_band,
            out_band,
        )
        return knut.export_gee_image_collection_connection(
            result_ic, ic_primary_connection
        )


############################################
# Image Collection Covariance/Correlation
############################################


@knext.node(
    name="GEE Image Collection Covariance/Correlation",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ICCorrelation.png",
    id="imagecollectioncovariancecorrelation",
    after="imagecollectiontimeseriesregressor",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="Image Collection where each image has at least two bands (e.g. current and lagged band for autocorrelation).",
    port_type=gee_image_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="Single image with per-pixel covariance and/or correlation band(s) across the collection.",
    port_type=gee_image_port_type,
)
class ICCovarianceCorrelation:
    """Computes per-pixel covariance or correlation between two bands across a collection.

    This node computes per-pixel covariance or correlation across time using two band names and output selection, and is commonly used for auto- or cross-correlation analysis.

    Reduces the collection in the time (image) dimension with ``ee.Reducer.covariance()``,
    then derives correlation as cov / sqrt(var1*var2). Use with two bands (e.g. NDVI and
    NDVI_1 for autocorrelation, or NDVI and precip_1 for cross-correlation).
    """

    band1_name = knext.StringParameter(
        "First band name",
        "Name of the first variable (e.g. 'NDVI' for autocorrelation, or 'NDVI' for cross with second band).",
        default_value="NDVI",
    )

    band2_name = knext.StringParameter(
        "Second band name",
        "Name of the second variable (e.g. 'NDVI_1' for lag-1 autocorrelation, or 'precip_1' for cross).",
        default_value="NDVI_1",
    )

    output_mode = knext.StringParameter(
        "Output",
        "Output bands: covariance only, correlation only, or both.",
        default_value="both",
        enum=["covariance", "correlation", "both"],
    )

    parallel_scale = knext.IntParameter(
        "Parallel scale",
        "Scale factor for reduce (increase if computation runs out of memory).",
        default_value=8,
        min_value=1,
        max_value=64,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, ic_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        ic = ic_connection.image_collection
        b1 = (self.band1_name or "").strip()
        b2 = (self.band2_name or "").strip()
        if not b1 or not b2:
            raise ValueError("Both band names are required.")
        if b1 == b2:
            raise ValueError("First and second band names must differ.")

        # Match book: map each image to array (1 band = 2-element vector), then reduce covariance
        ic_two = ic.select([b1, b2]).map(lambda img: img.toArray())
        reducer = ee.Reducer.covariance()
        cov_array_image = ic_two.reduce(reducer, parallelScale=self.parallel_scale)
        # 2x2 matrix: [0,0]=var1, [1,1]=var2, [0,1]=[1,0]=cov; arrayGet returns scalar band(s)
        var1 = cov_array_image.arrayGet([0, 0]).select([0])
        var2 = cov_array_image.arrayGet([1, 1]).select([0])
        cov = (
            cov_array_image.arrayGet([0, 1]).select([0]).rename("covariance").toFloat()
        )

        out_images = []
        if self.output_mode in ("covariance", "both"):
            out_images.append(cov)
        if self.output_mode in ("correlation", "both"):
            sqrt_prod = var1.multiply(var2).max(1e-18).sqrt()
            corr = cov.divide(sqrt_prod).rename("correlation").toFloat()
            out_images.append(corr)

        if not out_images:
            raise ValueError("Output mode must produce at least one band.")
        result = out_images[0]
        for img in out_images[1:]:
            result = result.addBands(img)
        first_proj = ic.first().select(b1).projection()
        scale_m = ic.first().select(b1).projection().nominalScale()
        result = result.reproject(crs=first_proj, scale=scale_m)
        LOGGER.warning(
            "Covariance/Correlation: bands (%s, %s) -> %s; reproject scale=nominalScale (match input resolution)",
            b1,
            b2,
            self.output_mode,
        )
        return knut.export_gee_image_connection(result, ic_connection)
