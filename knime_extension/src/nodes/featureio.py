"""
GEE Feature Collection I/O Nodes for KNIME
This module contains nodes for reading, filtering, clipping, and processing Google Earth Engine Feature Collections.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

# Category for GEE Feature Collection I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="featureio",
    name="Feature Collection IO",
    description="Google Earth Engine Feature Collection Input/Output and Processing nodes",
    icon="icons/featureIO.png",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/feature/"


############################################
# GEE Feature Collection Reader
############################################


@knext.node(
    name="Feature Collection Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "fcreader.png",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=google_earth_engine_port_type,
)
class GEEFeatureCollectionReader:
    """Loads a feature collection from Google Earth Engine using the specified collection ID.

    This node allows you to access vector datasets like administrative boundaries,
    points of interest, or other geospatial vector data from GEE's extensive catalog.
    The node outputs a GEE Feature Collection connection object for further processing.

    **Common Feature Collections:**

    - **GAUL Administrative Units**: 'FAO/GAUL/2015/level0' (country boundaries)

    - **GAUL Level 1**: 'FAO/GAUL/2015/level1' (state/province boundaries)

    - **GAUL Level 2**: 'FAO/GAUL/2015/level2' (county/district boundaries)

    - **World Countries**: 'USDOS/LSIB_SIMPLE/2017' (country boundaries)

    - **US States**: 'TIGER/2018/States' (US state boundaries)

    - **Protected Areas**: 'WCMC/WDPA/current/polygons' (protected areas)

    - **Cities**: 'USDOS/LSIB_SIMPLE/2017' (city boundaries)
    """

    collection_id = knext.StringParameter(
        "Collection ID",
        "The ID of the GEE feature collection (e.g., 'FAO/GAUL/2015/level0')",
        default_value="FAO/GAUL/2015/level0",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # GEE is already initialized in the same Python process from the connection

        # Load feature collection
        feature_collection = ee.FeatureCollection(self.collection_id)

        try:
            LOGGER.warning(f"Loaded feature collection: {self.collection_id}")
            return knut.export_gee_connection(feature_collection, gee_connection)

        except Exception as e:
            LOGGER.error(f"Failed to load feature collection: {e}")
            raise


############################################
# Feature Collection to Table
############################################
@knext.node(
    name="Feature Collection to Table",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "fc2table.png",
    after="",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_table(
    name="Output Table",
    description="Table converted from GEE Feature Collection",
)
class FeatureCollectionToTable:
    """Converts a Google Earth Engine FeatureCollection to a local table.

    This node converts a Google Earth Engine FeatureCollection to a KNIME table,
    allowing you to work with GEE vector data in standard tabular format.
    This node bridges GEE vector operations with KNIME's data processing capabilities,
    making it useful for exporting classification results, converting GEE vector analysis outputs,
    and processing GEE-generated point samples or administrative boundaries.

    **Output Formats:**

    - **DataFrame**: Standard tabular format with attribute data only

    - **GeoDataFrame**: Tabular format with embedded geometry information

    **Note:** Data transfer from Google Earth Engine cloud to local systems is subject to GEE's transmission limits.
    For large FeatureCollections, using loop to process the data is recommended.
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
        fc_connection,
    ):
        import ee
        import logging
        import pandas as pd
        import geemap

        # LOGGER = logging.getLogger(__name__)

        # Get feature collection directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        feature_collection = fc_connection.gee_object

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


############################################
# GeoTable to Feature Collection
############################################


@knext.node(
    name="GeoTable to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "table2fc.png",
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
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=google_earth_engine_port_type,
)
class GeoTableToFeatureCollection:
    """Converts a local GeoTable to a Google Earth Engine FeatureCollection.

    This node converts a KNIME table containing geometry data to a Google Earth Engine FeatureCollection,
    enabling vector data processing in GEE workflows. This node bridges local GIS data with GEE's processing capabilities,
    making it useful for uploading study area boundaries, converting training samples for classification,
    processing custom administrative boundaries, and working with field survey data or sampling points.

    **Note:** Data transfer from local systems to Google Earth Engine cloud is subject to GEE's transmission limits.
    For large geometry datasets, consider processing in smaller batches to avoid data limit errors.

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

        # GEE is already initialized in the same Python process from the connection

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

        return knut.export_gee_connection(feature_collection, gee_connection)


############################################
# GEE Feature Collection Filter
############################################


@knext.node(
    name="Feature Collection Value Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "valueFilter.png",
    after="",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with filtered feature collection object.",
    port_type=google_earth_engine_port_type,
)
class GEEFeatureCollectionFilter:
    """Filters a Google Earth Engine Feature Collection based on property values with advanced comparison operators.

    This node allows you to filter Feature Collections using various comparison operators (equals, greater than,
    contains, etc.) on feature properties. This node is useful for extracting specific subsets of large Feature Collections,
    reducing data size for processing, and focusing analysis on specific administrative units.

    **Filter Operators:**

    - **Equals/Not Equals**: Exact value matching
    - **Greater/Less Than**: Numeric comparisons
    - **String Operations**: Contains, starts with, ends with
    - **List Operations**: Match any value in a comma-separated list

    **Common Use Cases:**

    - Extract specific countries from global administrative boundaries

    - Filter protected areas by type or status

    - Filter by numeric properties (e.g., administrative levels, IDs)

    - Complex string pattern matching

    - Multi-value filtering

    **Note:** For spatial filtering, use the "Feature Collection Spatial Filter" node.
    """

    # Define all available operators
    OPERATOR_CHOICES = {
        "Equals": "eq",
        "Not Equals": "neq",
        "Greater Than": "gt",
        "Greater Than or Equals": "gte",
        "Less Than": "lt",
        "Less Than or Equals": "lte",
        "String Contains": "stringContains",
        "String Starts With": "stringStartsWith",
        "String Ends With": "stringEndsWith",
        "Is In List (comma-separated)": "inList",
    }

    # Create operator dropdown parameter
    filter_operator = knext.StringParameter(
        "Filter Operator",
        "The comparison operator to use for filtering.",
        default_value="Equals",
        enum=list(OPERATOR_CHOICES.keys()),
    )

    property_name = knext.StringParameter(
        "Property Name",
        "Name of the property to filter by (e.g., 'ADM0_NAME').",
        default_value="",
    )

    property_value = knext.StringParameter(
        "Property Value",
        "Value to filter by. For 'Is In List', use comma-separated values (e.g., 'China,Japan,India').",
        default_value="",
    )

    max_features = knext.IntParameter(
        "Maximum Features",
        "Maximum number of features to return (-1 = no limit).",
        default_value=-1,
        min_value=-1,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, fc_connection):
        import ee
        import logging
        import util.knime_utils as knut

        LOGGER = logging.getLogger(__name__)
        feature_collection = fc_connection.gee_object

        if self.property_name and self.property_value:
            op_key = self.OPERATOR_CHOICES[self.filter_operator]
            prop = self.property_name
            val_str = str(self.property_value).strip()

            val_num = ee.Number.parse(val_str)
            eq_num = ee.Filter.eq(prop, val_num)
            eq_str = ee.Filter.eq(prop, val_str)
            eq_any = ee.Filter.Or(eq_num, eq_str)

            if op_key == "eq":  # Equals
                the_filter = eq_any

            elif op_key == "neq":  # Not Equals
                # Not(eq_num OR eq_str OR eq_str_lower)
                try:
                    the_filter = ee.Filter.Not(eq_any)
                except Exception:
                    # not_；
                    the_filter = ee.Filter.not_(eq_any)

            elif op_key in ["gt", "gte", "lt", "lte"]:

                the_filter = {
                    "gt": ee.Filter.gt,
                    "gte": ee.Filter.gte,
                    "lt": ee.Filter.lt,
                    "lte": ee.Filter.lte,
                }[op_key](prop, val_num)

            elif op_key == "stringContains":
                the_filter = ee.Filter.stringContains(prop, val_str)
            elif op_key == "stringStartsWith":
                the_filter = ee.Filter.stringStartsWith(prop, val_str)
            elif op_key == "stringEndsWith":
                the_filter = ee.Filter.stringEndsWith(prop, val_str)

            elif op_key == "inList":
                str_list = ee.List(
                    [s.strip() for s in val_str.split(",") if s.strip() != ""]
                )
                num_list = str_list.map(lambda s: ee.Number.parse(s))
                in_str = ee.Filter.inList(prop, str_list)
                in_num = ee.Filter.inList(prop, num_list)
                the_filter = ee.Filter.Or(in_str, in_num)

            else:
                the_filter = None

            if the_filter:
                LOGGER.warning(f"Applying filter: {prop} {op_key} {val_str}")
                feature_collection = feature_collection.filter(the_filter)

        if self.max_features >= 0:
            feature_collection = feature_collection.limit(self.max_features)

        # Check if the result is empty and provide warning
        try:
            result_size = feature_collection.size()
            if result_size.getInfo() == 0:
                LOGGER.warning(
                    "Property filter operation resulted in empty FeatureCollection"
                )
        except Exception as e:
            LOGGER.warning(f"Could not check result size: {e}")

        return knut.export_gee_connection(feature_collection, fc_connection)


############################################
# GEE Feature Collection Spatial Filter
############################################


@knext.node(
    name="Feature Collection Spatial Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "spatialFilter.png",
    after="",
)
@knext.input_port(
    name="Input Feature Collection",
    description="The Feature Collection to be filtered or clipped.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="Filter Feature Collection",
    description="The Feature Collection used as the spatial filter or clipping boundary.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="The resulting spatially filtered or clipped Feature Collection.",
    port_type=google_earth_engine_port_type,
)
class GEEFeatureCollectionSpatialFilter:
    """
    Performs powerful spatial filtering and clipping on Feature Collections based on precise geometric relationships.

    This node provides comprehensive spatial analysis capabilities for Feature Collections, supporting various
    geometric relationships and operations. It is useful for spatial data analysis, geographic filtering,
    and geometric operations on vector data.

    **Spatial Operations:**

    - **Intersects**: Find features that spatially intersect with the filter geometry

    - **Contains**: Find features that completely contain the filter geometry

    - **Within**: Find features that are completely within the filter geometry

    - **Disjoint**: Find features that are completely separate from the filter geometry

    - **Within Distance**: Find features within a specified distance of the filter geometry

    - **Clip to Shape**: Clip features to the exact shape of the filter geometry

    - **Intersects Bounding Box**: Fast but less precise filtering using bounding boxes

    **Common Use Cases:**

    - Extract features within study areas or administrative boundaries

    - Find features near roads, cities, or other reference features


    - Perform spatial analysis and geographic filtering

    - Remove or isolate features based on spatial relationships

    **Performance Notes:**

    - **Intersects Bounding Box**: Fastest but less precise
    - **Intersects**: Precise but slower
    """

    # Define all available spatial operations
    OPERATOR_CHOICES = {
        "Intersects": "intersects",
        "Contains": "contains",
        "Within": "within",
        "Disjoint": "disjoint",
        "Within Distance": "withinDistance",
        "Intersects Bounding Box": "filterBounds",
    }

    # Create spatial operator dropdown
    spatial_operator = knext.StringParameter(
        "Spatial Operator",
        "The spatial relationship or operation to apply.",
        default_value="Intersects",
        enum=list(OPERATOR_CHOICES.keys()),
    )

    # Add distance parameter for "Within Distance" operation
    distance = knext.DoubleParameter(
        "Distance (meters)",
        "The distance in meters for the 'Within Distance' operator. Ignored by other operators.",
        default_value=1000.0,
        min_value=0.0,
    ).rule(knext.OneOf(spatial_operator, ["Within Distance"]), knext.Effect.SHOW)

    # Add option for "Clip" operation

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, fc_connection, filter_fc_connection
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        feature_collection = fc_connection.gee_object
        filter_geometry = filter_fc_connection.gee_object.geometry()

        op_key = self.OPERATOR_CHOICES[self.spatial_operator]

        result_fc = None
        LOGGER.warning(f"Applying spatial operator: {op_key}")

        # Core logic: Execute different code based on selected operation
        # All operations belong to filter
        the_filter = None
        # This is a common left-side definition for spatial filters
        left_field = ".geo"

        if op_key == "filterBounds":
            # .bounds() takes the geometry directly
            the_filter = ee.Filter.bounds(filter_geometry)
        elif op_key == "intersects":
            # Correct function name is .intersects()
            the_filter = ee.Filter.intersects(
                leftField=left_field, rightValue=filter_geometry
            )
        elif op_key == "contains":
            # Using the more robust syntax for consistency
            the_filter = ee.Filter.contains(
                leftField=left_field, rightValue=filter_geometry
            )
        elif op_key == "within":
            # "within" is equivalent to "withinDistance" of 0 meters
            the_filter = ee.Filter.withinDistance(
                leftField=left_field, rightValue=filter_geometry, distance=0
            )
        elif op_key == "disjoint":
            # Using the more robust syntax for consistency
            the_filter = ee.Filter.disjoint(
                leftField=left_field, rightValue=filter_geometry
            )
        elif op_key == "withinDistance":
            the_filter = ee.Filter.withinDistance(
                leftField=left_field,
                rightValue=filter_geometry,
                distance=self.distance,
            )

        if the_filter:
            result_fc = feature_collection.filter(the_filter)
        else:
            # If no matching operation, return original collection to avoid errors
            result_fc = feature_collection

        # Check if the result is empty and provide warning
        try:
            result_size = result_fc.size()
            if result_size.getInfo() == 0:
                LOGGER.warning(
                    "Spatial filter operation resulted in empty FeatureCollection"
                )
        except Exception as e:
            LOGGER.warning(f"Could not check result size: {e}")

        return knut.export_gee_connection(result_fc, fc_connection)
