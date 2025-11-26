"""
GEE Feature Collection I/O Nodes for KNIME
This module contains nodes for reading, filtering, clipping, and processing Google Earth Engine Feature Collections.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_feature_collection_port_type,
)

# Category for GEE Feature Collection I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="featureio",
    name="Feature Collection IO",
    description="Google Earth Engine Feature Collection Input/Output and Processing nodes",
    icon="icons/featureIO.png",
    after="imageio",
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
    id="fcreader",
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
    port_type=gee_feature_collection_port_type,
)
class GEEFeatureCollectionReader:
    """Loads a feature collection from Google Earth Engine using the specified collection ID.

    This node allows you to access vector datasets like administrative boundaries,
    points of interest, or other geospatial vector data from GEE's extensive catalog.
    The node outputs a GEE Feature Collection connection object for further processing.

    Visit [GEE Datasets Catalog](https://developers.google.com/earth-engine/datasets/catalog) to explore available
    datasets and their image IDs.

    **Common Feature Collections:**

    - [GAUL Administrative Units](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level0): 'FAO/GAUL/2015/level0' (country boundaries)
    - [GAUL Level 1](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level1): 'FAO/GAUL/2015/level1' (state/province boundaries)
    - [GAUL Level 2](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level2): 'FAO/GAUL/2015/level2' (county/district boundaries)
    - [World Countries](https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017): 'USDOS/LSIB_SIMPLE/2017' (country boundaries)
    - [US States](https://developers.google.com/earth-engine/datasets/catalog/TIGER_2018_States): 'TIGER/2018/States' (US state boundaries)
    - [Protected Areas](https://developers.google.com/earth-engine/datasets/catalog/WCMC_WDPA_current_polygons): 'WCMC/WDPA/current/polygons' (protected areas)
    - [Cities](https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017): 'USDOS/LSIB_SIMPLE/2017' (city boundaries)

    **Note:** The GEE Dataset Search node can help you find more feature collections.
    """

    collection_id = knext.StringParameter(
        "Collection ID",
        """The ID of the GEE feature collection (e.g., 'FAO/GAUL/2015/level0'). 
        You can use the GEE Dataset Search node to find available collections or 
        visit [GEE Datasets Catalog](https://developers.google.com/earth-engine/datasets/catalog).""",
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
            return knut.export_gee_feature_collection_connection(
                feature_collection, gee_connection
            )

        except Exception as e:
            LOGGER.error(f"Failed to load feature collection: {e}")
            raise


############################################
# GEE Feature Collection Filter
############################################


@knext.node(
    name="Feature Collection Value Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "valueFilter.png",
    id="valuefilter",
    after="fcreader",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with filtered feature collection object.",
    port_type=gee_feature_collection_port_type,
)
class GEEFeatureCollectionFilter:
    """Filters a Google Earth Engine Feature Collection based on property values with advanced comparison operators.

    This node allows you to filter Feature Collections using various comparison operators (equals, greater than,
    contains, etc.) on feature properties. This node is useful for extracting specific subsets of large
    Feature Collections, reducing data size for processing, and focusing analysis on specific administrative units.

    To get a list of available properties, use the "Feature Collection Info" node.

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
        """Name of the property to filter by (e.g., 'ADM0_NAME'). To get a list of available properties, 
        use the "Feature Collection Info" node.""",
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

        LOGGER = logging.getLogger(__name__)
        feature_collection = ee.FeatureCollection(fc_connection.feature_collection)

        # Only proceed if both property name and value are set
        prop = (self.property_name or "").strip()
        val_str = (self.property_value or "").strip()

        if prop and val_str:
            # Client-side check: is the value numeric?
            def _to_float_or_none(s):
                try:
                    v = float(s)
                    # Avoid NaN/Inf
                    if v == float("inf") or v == float("-inf") or v != v:
                        return None
                    return v
                except Exception:
                    return None

            val_float = _to_float_or_none(val_str)
            has_numeric = val_float is not None

            # Filter out features with missing property to avoid comparison errors
            feature_collection = feature_collection.filter(ee.Filter.notNull([prop]))

            op_key = self.OPERATOR_CHOICES[self.filter_operator]
            the_filter = None

            # ============== Equals / Not Equals ==============
            if op_key in ("eq", "neq"):
                filters = []
                # String equality
                filters.append(ee.Filter.eq(prop, val_str))
                # Numeric equality (only if value is actually numeric)
                if has_numeric:
                    filters.append(ee.Filter.eq(prop, ee.Number(val_float)))

                if len(filters) == 1:
                    eq_any = filters[0]
                else:
                    eq_any = ee.Filter.Or(filters[0], filters[1])

                if op_key == "eq":
                    the_filter = eq_any
                else:
                    # Not(eq_any)
                    try:
                        the_filter = ee.Filter.not_(eq_any)
                    except Exception:
                        the_filter = ee.Filter.Not(eq_any)

            # ============== Numeric Comparisons ==============
            elif op_key in ("gt", "gte", "lt", "lte"):
                if not has_numeric:
                    LOGGER.warning(
                        f"Numeric operator '{op_key}' requires a numeric value, but got '{val_str}'. "
                        "This filter will be skipped."
                    )
                else:
                    cmp_func = {
                        "gt": ee.Filter.gt,
                        "gte": ee.Filter.gte,
                        "lt": ee.Filter.lt,
                        "lte": ee.Filter.lte,
                    }[op_key]
                    the_filter = cmp_func(prop, ee.Number(val_float))

            # ============== String Operations ==============
            elif op_key == "stringContains":
                the_filter = ee.Filter.stringContains(prop, val_str)
            elif op_key == "stringStartsWith":
                the_filter = ee.Filter.stringStartsWith(prop, val_str)
            elif op_key == "stringEndsWith":
                the_filter = ee.Filter.stringEndsWith(prop, val_str)

            # ============== inList (compatible with both strings and numbers) ==============
            elif op_key == "inList":
                import re

                # 1) Normalize: remove spaces and quotes, but preserve original string form (leading zeros, etc.)
                tokens = [
                    t.strip().strip('"').strip("'")
                    for t in val_str.split(",")
                    if t.strip()
                ]

                # 2) Always keep "string version" of the list (for string attributes / preserve leading zeros)
                str_tokens = tokens

                # 3) Only for "strict numeric" tokens, additionally build numeric list (for numeric attributes)
                #    Strict numeric: can contain +/- and decimal point, but no leading/trailing whitespace or non-numeric chars
                def _is_strict_numeric(s: str) -> bool:
                    return re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s) is not None

                num_tokens = [float(t) for t in tokens if _is_strict_numeric(t)]

                sub_filters = []
                if str_tokens:
                    sub_filters.append(ee.Filter.inList(prop, ee.List(str_tokens)))
                if num_tokens:
                    sub_filters.append(ee.Filter.inList(prop, ee.List(num_tokens)))

                if len(sub_filters) == 1:
                    the_filter = sub_filters[0]
                elif len(sub_filters) == 2:
                    the_filter = ee.Filter.Or(sub_filters[0], sub_filters[1])

            # Apply filter only if we successfully built one
            if the_filter:
                LOGGER.warning(f"Applying filter: {prop} {op_key} {val_str}")
                feature_collection = feature_collection.filter(the_filter)
            else:
                LOGGER.warning(
                    f"No valid filter built for: {prop} {op_key} {val_str}; skipping value filter."
                )

        # Limit number of features
        if self.max_features >= 0:
            try:
                feature_collection = feature_collection.limit(int(self.max_features))
            except Exception as ex:
                LOGGER.warning(f"Limit failed and was skipped: {ex}")

        # Check result size (triggers server-side evaluation)
        try:
            result_size = feature_collection.size().getInfo()
            if result_size == 0:
                LOGGER.warning(
                    "Property filter operation resulted in empty FeatureCollection"
                )
            else:
                LOGGER.warning(f"Filtered FeatureCollection size: {result_size}")
        except Exception as e:
            LOGGER.warning(f"Could not check result size: {e}")

        return knut.export_gee_feature_collection_connection(
            feature_collection, fc_connection
        )


############################################
# GEE Feature Collection Spatial Filter
############################################


@knext.node(
    name="Feature Collection Spatial Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "spatialFilter.png",
    id="spatialfilter",
    after="valuefilter",
)
@knext.input_port(
    name="Input Feature Collection",
    description="The Feature Collection to be filtered or clipped.",
    port_type=gee_feature_collection_port_type,
)
@knext.input_port(
    name="Filter Feature Collection",
    description="The Feature Collection used as the spatial filter or clipping boundary.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="The resulting spatially filtered or clipped Feature Collection.",
    port_type=gee_feature_collection_port_type,
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

    **Note:** For value based filtering, use the "Feature Collection Value Filter" node.
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
        feature_collection = fc_connection.feature_collection
        filter_geometry = filter_fc_connection.feature_collection.geometry()

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

        return knut.export_gee_feature_collection_connection(result_fc, fc_connection)


############################################
# Feature Collection Info
############################################


@knext.node(
    name="Feature Collection Info",
    node_type=knext.NodeType.VISUALIZER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "fcinfo.png",
    id="fcinfo",
    after="spatialfilter",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_table(
    name="Feature Collection Info Table",
    description="Table containing property names, types, and geometry information",
)
class GEEFeatureCollectionInfo:
    """Displays property information about a Google Earth Engine Feature Collection in table format.

    This node extracts and displays property names, their data types, and geometry type
    from a Feature Collection in a structured table format. This is useful for data
    exploration, understanding the structure of your vector data, and selecting
    appropriate properties for further processing.

    **Information Displayed:**

    - **Property Names**: All available property/attribute names
    - **Property Types**: Data types of each property (string, number, boolean, etc.)
    - **Geometry Type**: Type of geometries in the collection

    **Use Cases:**

    - Explore and understand new datasets
    - Verify data structure before analysis
    - Select appropriate properties for filtering or visualization
    - Check data quality and completeness
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)
        feature_collection = fc_connection.feature_collection

        try:
            # --- Optimized Server-Side Analysis ---

            # 1. Get a reference to the first feature without downloading it
            first_feature = feature_collection.first()

            # 2. Get property names and geometry type as server-side objects
            prop_names = first_feature.propertyNames()
            geom_type = first_feature.geometry().type()

            # 3. Define a server-side function to get the type of a property
            def get_prop_type(prop_name):
                prop_value = first_feature.get(prop_name)
                # ee.Algorithms.ObjectType efficiently gets the type as a string
                return ee.Algorithms.ObjectType(prop_value)

            # 4. Map the function over the property names to get a list of types
            prop_types = prop_names.map(get_prop_type)

            # 5. Combine names and types into a server-side dictionary
            properties_info = ee.Dictionary.fromLists(prop_names, prop_types)

            # 6. Combine everything into a final dictionary for one single download
            all_info = ee.Dictionary(
                {"properties": properties_info, "geometry_type": geom_type}
            )

            # 7. Make ONE fast getInfo() call to download the small summary
            result_info = all_info.getInfo()

            # --- Client-Side Table Creation (no changes needed here) ---

            property_data = []
            if "properties" in result_info and result_info["properties"]:
                # The property types are already strings from ee.Algorithms.ObjectType
                for prop_name, prop_type in result_info["properties"].items():
                    property_data.append(
                        {
                            "Property Name": prop_name,
                            "Property Type": prop_type.lower(),  # e.g., 'Number', 'String'
                        }
                    )
            else:
                property_data.append(
                    {
                        "Property Name": "No properties found",
                        "Property Type": "N/A",
                    }
                )

            # Add geometry type as a separate row
            geometry_type = result_info.get("geometry_type", "Unknown")
            property_data.append(
                {
                    "Property Name": "geometry",
                    "Property Type": geometry_type,
                }
            )

            df = pd.DataFrame(property_data)
            return knext.Table.from_pandas(df)

        except Exception as e:
            # Handle cases where the collection might be empty
            LOGGER.error(f"Failed to get Feature Collection info: {e}")
            df = pd.DataFrame(
                [
                    {
                        "Property Name": "Error",
                        "Property Type": "Could not retrieve info. Collection might be empty or invalid.",
                    }
                ]
            )
            return knext.Table.from_pandas(df)


############################################
# Feature Collection to Table
############################################
@knext.node(
    name="Feature Collection to Table",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "fc2table.png",
    id="fc2table",
    after="fcinfo",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=gee_feature_collection_port_type,
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

    **⚠️ IMPORTANT - Data Size Limitations:**

    This node is designed for **small to medium-sized Feature Collections** only.
    It uses GEE's interactive API which has a **10MB payload limit**.

    **Recommended Use Cases:**
    - Small collections (< 1000 features with simple attributes)
    - Quick data previews and exploration
    - Testing and debugging workflows

    **For Large Datasets:**
    - Use **"Feature Collection Exporter"** node for large collections
    - Export uses GEE's batch processing system (no payload limits)
    - Suitable for production workflows with millions of features

    **Output Formats:**

    - **DataFrame**: Standard tabular format with attribute data only
    - **GeoDataFrame**: Tabular format with embedded geometry information

    **When to Use Export Instead:**

    If you encounter errors like "Request payload size exceeds the limit",
    your Feature Collection is too large for direct conversion. Please use
    the "Feature Collection Exporter" node instead.
    """

    file_format = knext.StringParameter(
        "Output Format",
        """Format for the output table. 
        **Output Formats:**
        
        - **DataFrame**: Standard tabular format with attribute data only
        - **GeoDataFrame**: Tabular format with embedded geometry information
        """,
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

        LOGGER = logging.getLogger(__name__)

        # Get feature collection directly from connection object
        feature_collection = fc_connection.feature_collection

        try:
            LOGGER.warning(
                "Converting Feature Collection to table (for small collections only)"
            )

            # Direct conversion - no batch processing
            if self.file_format == "DataFrame":
                df = geemap.ee_to_df(feature_collection)
            else:  # GeoDataFrame
                df = geemap.ee_to_gdf(feature_collection)

            # Remove RowID column if present
            if "<RowID>" in df.columns:
                df = df.drop(columns=["<RowID>"])

            LOGGER.warning(
                f"Successfully converted Feature Collection to table with {len(df)} rows"
            )

            return knext.Table.from_pandas(df)

        except Exception as e:
            error_msg = str(e).lower()
            if "payload" in error_msg or "limit" in error_msg:
                detailed_error = (
                    f"Feature Collection is too large for direct conversion.\n\n"
                    f"Error: {e}\n\n"
                    f"Solution: Use the 'Feature Collection to Drive' node instead.\n"
                    f"The Export node uses GEE's batch processing system and can handle "
                    f"much larger datasets without payload limits."
                )
                LOGGER.error(detailed_error)
                raise ValueError(detailed_error)
            else:
                LOGGER.error(f"Feature Collection to Table conversion failed: {e}")
                raise


############################################
# GeoTable to Feature Collection
############################################


@knext.node(
    name="GeoTable to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "table2fc.png",
    id="table2fc",
    after="fc2table",
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
    port_type=gee_feature_collection_port_type,
)
class GeoTableToFeatureCollection:
    """Converts a local GeoTable to a Google Earth Engine FeatureCollection.

    This node converts a KNIME table containing geometry data to a Google Earth Engine FeatureCollection,
    enabling vector data processing in GEE workflows. This node bridges local GIS data with GEE's processing capabilities,
    making it useful for uploading study area boundaries, converting training samples for classification,
    processing custom administrative boundaries, and working with field survey data or sampling points.

    **Batch Processing:**

    For large GeoTables, batch processing is automatically enabled to handle
    GEE's upload limits. The node processes features in batches and combines
    them into a single Feature Collection automatically.

    **Note:** Data transfer from local systems to Google Earth Engine cloud is subject to GEE's transmission limits.
    Batch processing helps avoid these limits for large datasets.

    """

    geo_col = knext.ColumnParameter(
        "Geometry Column",
        "Column containing geometry data",
        port_index=0,
    )

    batch_size = knext.IntParameter(
        "Batch Size",
        "Number of features to process in each batch (smaller batches = safer but slower). Batch processing is automatically enabled for large tables.",
        default_value=500,
        min_value=50,
        max_value=5000,
        is_advanced=True,
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

        # Remove <RowID> and index columns to avoid duplicate column names
        if "<RowID>" in df.columns:
            df = df.drop(columns=["<RowID>"])
        # Reset index to avoid index column issues
        df = df.reset_index(drop=True)

        LOGGER.warning(f"Converting table with {len(df)} rows to Feature Collection")

        # Create GeoDataFrame
        shp = gp.GeoDataFrame(df, geometry=self.geo_col)

        # Ensure CRS is WGS84 (EPSG:4326)
        if shp.crs is None:
            shp.set_crs(epsg=4326, inplace=True)
        else:
            shp.to_crs(4326, inplace=True)

        LOGGER.warning(f"GeoDataFrame CRS: {shp.crs}")

        # Use batch processing if table is large (no need for enable_batch parameter)
        use_batch = len(shp) > 1000

        try:
            if use_batch:
                LOGGER.warning(
                    f"Using batch processing with batch size {self.batch_size}"
                )
                feature_collection = (
                    knut.batch_process_geodataframe_to_feature_collection(
                        shp,
                        batch_size=self.batch_size,
                        logger=LOGGER,
                        exec_context=exec_context,
                    )
                )
            else:
                # Direct conversion for small tables
                LOGGER.warning("Converting GeoDataFrame directly to Feature Collection")
                feature_collection = geemap.gdf_to_ee(shp)

            LOGGER.warning(
                f"Successfully converted {len(shp)} features to Feature Collection"
            )
            return knut.export_gee_feature_collection_connection(
                feature_collection, gee_connection
            )

        except Exception as e:
            # If direct conversion fails (likely due to size), try batch processing
            if not use_batch:
                LOGGER.warning(
                    f"Direct conversion failed ({e}), retrying with batch processing"
                )
                try:
                    feature_collection = (
                        knut.batch_process_geodataframe_to_feature_collection(
                            shp,
                            batch_size=self.batch_size,
                            logger=LOGGER,
                            exec_context=exec_context,
                        )
                    )
                    LOGGER.warning(
                        f"Successfully converted {len(shp)} features to Feature Collection (using batch processing)"
                    )
                    return knut.export_gee_connection(
                        feature_collection, gee_connection
                    )
                except Exception as batch_error:
                    LOGGER.error(f"Batch processing also failed: {batch_error}")
                    raise
            else:
                LOGGER.error(f"GeoTable to Feature Collection conversion failed: {e}")
                raise


############################################
# Feature Collection Exporter
############################################


@knext.node(
    name="Feature Collection Exporter",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "Feature2Cloud.png",
    id="fcexporter",
    after="fc2table",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionExporter:
    """Exports a FeatureCollection to Google Drive or Google Cloud Storage.

    **Authentication & Scopes**

    - Always include ``https://www.googleapis.com/auth/earthengine`` in the Google Authenticator node.
    - *Drive exports* additionally require a Drive scope such as ``https://www.googleapis.com/auth/drive``.
    - *Cloud exports* additionally require ``https://www.googleapis.com/auth/cloud-platform``.
    - When authenticating with a **Service Account**, add the cloud-platform scope; Drive export is not supported for service accounts.
    - When using **Interactive Authentication**, you can add both scopes and choose either destination.

    **Destination Path**

    - Drive: ``DriveFolder/file_name`` (file extension appended automatically).
    - Cloud Storage: ``bucket/path/file_name`` (object created directly under the bucket path).

    **Export Formats & Files**

    - ``CSV`` → Creates a single ``.csv`` file (geometry as WKT).
    - ``GeoJSON`` → Creates a single ``.geojson`` file (full geometry).
    - ``KML`` → Creates a ``.kml`` file.
    - ``KMZ`` → Creates a compressed ``.kmz`` file.
    - ``SHP`` → Generates multiple shapefile components (``.shp``, ``.shx``, ``.dbf``, ``.prj``) in the chosen destination; consider using CSV/GeoJSON for simplicity.

    Enable "Wait for Completion" to block until the export finishes (with a configurable timeout).
    """

    class DestinationModeOptions(knext.EnumParameterOptions):
        CLOUD = (
            "Google Cloud Storage",
            "Export FeatureCollection to a Google Cloud Storage bucket.",
        )
        DRIVE = ("Google Drive", "Export FeatureCollection to a Google Drive folder.")

        @classmethod
        def get_default(cls):
            return cls.CLOUD

    destination = knext.EnumParameter(
        label="Destination Mode",
        description="Select export destination. Drive requires interactive authentication with Drive scope.",
        default_value=DestinationModeOptions.get_default().name,
        enum=DestinationModeOptions,
    )

    export_format = knext.StringParameter(
        "Export Format",
        """Format for the exported file. Available formats:
        
        - ``CSV`` → Creates a single ``.csv`` file (geometry as WKT).
        - ``GeoJSON`` → Creates a single ``.geojson`` file (full geometry).
        - ``KML`` → Creates a ``.kml`` file.
        - ``KMZ`` → Creates a compressed ``.kmz`` file.
        - ``SHP`` → Generates multiple shapefile components (``.shp``, ``.shx``, ``.dbf``, ``.prj``) in the chosen 
        destination; consider using CSV/GeoJSON for simplicity.
        """,
        default_value="CSV",
        enum=["CSV", "GeoJSON", "KML", "KMZ", "SHP"],
    )

    export_path = knext.StringParameter(
        "Destination Path",
        "Destination path (Drive: 'DriveFolder/file'; Cloud Storage: 'bucket/path/file').",
        default_value="EEexport/feature_collection_export",
    )

    wait_for_completion = knext.BoolParameter(
        "Wait for Completion",
        "If enabled, the node waits until the export task completes.",
        default_value=False,
    )

    max_wait_seconds = knext.IntParameter(
        "Max Wait Seconds",
        "Maximum number of seconds to wait when waiting is enabled.",
        default_value=600,
        min_value=1,
        is_advanced=True,
    ).rule(knext.OneOf(wait_for_completion, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging
        import time
        from datetime import datetime

        LOGGER = logging.getLogger(__name__)

        feature_collection = fc_connection.feature_collection

        destination = self.destination or self.DestinationModeOptions.get_default().name

        format_map = {
            "CSV": "CSV",
            "GeoJSON": "GeoJSON",
            "KML": "KML",
            "KMZ": "KMZ",
            "SHP": "SHP",
        }
        gee_format = format_map.get(self.export_format, "CSV")

        ext_map = {
            "CSV": ".csv",
            "GeoJSON": ".geojson",
            "KML": ".kml",
            "KMZ": ".kmz",
            "SHP": ".shp",
        }
        file_ext = ext_map.get(self.export_format, ".csv")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        description = f"KNIME Feature Collection Export {timestamp}"

        path = (self.export_path or "").strip()
        if not path:
            raise ValueError("Destination path is required.")

        try:
            destination_option = self.DestinationModeOptions[destination]
        except KeyError as exc:
            valid = ", ".join(opt.name for opt in self.DestinationModeOptions)
            raise ValueError(
                f"Unsupported destination '{self.destination}'. Choose one of [{valid}]."
            ) from exc

        if destination_option is self.DestinationModeOptions.DRIVE:
            if "/" in path:
                folder, prefix = path.rsplit("/", 1)
            else:
                folder, prefix = "EEexport", path

            LOGGER.warning(
                "Exporting FeatureCollection to Drive folder '%s' as '%s%s' (format: %s).",
                folder,
                prefix,
                file_ext,
                self.export_format,
            )

            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=description,
                folder=folder,
                fileNamePrefix=prefix,
                fileFormat=gee_format,
            )
        else:
            if not path.startswith("gs://"):
                path = f"gs://{path.lstrip('/')}"
            path = path.rstrip("/")

            if "/" not in path[5:]:
                raise ValueError(
                    "Cloud Storage path must include bucket and file name, e.g., 'bucket/path/file'."
                )

            bucket, object_prefix = path[5:].split("/", 1)

            LOGGER.warning(
                "Exporting FeatureCollection to Cloud Storage 'gs://%s/%s%s' (format: %s).",
                bucket,
                object_prefix,
                file_ext,
                self.export_format,
            )

            task = ee.batch.Export.table.toCloudStorage(
                collection=feature_collection,
                description=description,
                bucket=bucket,
                fileNamePrefix=object_prefix,
                fileFormat=gee_format,
            )

        task.start()
        LOGGER.warning("Export task started: %s", task.id)

        if not self.wait_for_completion:
            LOGGER.warning(
                "Export running in background. Monitor task status via GEE Code Editor or API."
            )
            return None

        LOGGER.warning("Waiting for export to complete...")
        start_time = time.time()
        max_wait = self.max_wait_seconds
        check_interval = 5

        while task.active():
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                LOGGER.error(
                    "Timed out waiting for export task %s after %s seconds.",
                    task.id,
                    max_wait,
                )
                break
            LOGGER.warning(
                "Export still running... elapsed %.1fs (max %ss).",
                elapsed,
                max_wait,
            )
            time.sleep(check_interval)

        status = task.status()
        state = status.get("state")
        if state == "COMPLETED":
            LOGGER.warning("Export completed successfully.")
        elif state == "FAILED":
            raise RuntimeError(
                f"Export task failed: {status.get('error_message', 'Unknown error')}"
            )
        else:
            LOGGER.warning("Export task ended with state: %s", state)

        return None


############################################
# Cloud Storage to Table
############################################


@knext.node(
    name="Cloud Storage to Table",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "Cloud2Table.png",
    id="cloudstorage2table",
    after="fcexporter",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node (used for authentication).",
    port_type=google_earth_engine_port_type,
)
@knext.output_table(
    name="Output Table",
    description="Table converted from Cloud Storage file (DataFrame or GeoDataFrame based on file format)",
)
class CloudStorageToTable:
    """Reads a file from Google Cloud Storage and converts it to a KNIME table.

    Provide the file as a single Cloud Storage path such as ``bucket/path/file.ext`` or
    ``gs://bucket/path/file.ext``. The bucket must already exist and the filename must include
    the desired extension so the loader can detect CSV, GeoJSON, KML/KMZ, or SHP formats.
    """

    cloud_path = knext.StringParameter(
        "Cloud Storage Path",
        "Path to the file in Google Cloud Storage (e.g., 'bucket/path/file.csv').",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        gee_connection,
    ):
        import logging
        import pandas as pd
        import geopandas as gpd
        import io
        from google.cloud import storage

        LOGGER = logging.getLogger(__name__)

        try:
            cloud_path = (self.cloud_path or "").strip()
            if not cloud_path:
                raise ValueError(
                    "Cloud Storage path is required (e.g., 'bucket/path/file.csv')."
                )

            if not cloud_path.startswith("gs://"):
                cloud_path = f"gs://{cloud_path.lstrip('/')}"

            LOGGER.warning(f"Reading file from Cloud Storage: {cloud_path}")

            # Get credentials from GEE connection
            credentials = gee_connection.credentials

            # Initialize Cloud Storage client
            # Use credentials from GEE connection (works for both service account and user credentials)
            try:
                # Try to refresh credentials if needed (for user credentials)
                if hasattr(credentials, "refresh") and credentials.expired:
                    credentials.refresh(None)
            except Exception:
                # If refresh fails, continue anyway (service account credentials don't need refresh)
                pass

            storage_client = storage.Client(
                credentials=credentials, project=gee_connection.spec.project_id
            )

            bucket_and_object = cloud_path[5:]
            if "/" not in bucket_and_object:
                raise ValueError(
                    "Cloud Storage path must include bucket and object, e.g., 'bucket/path/file.csv'."
                )

            bucket_name, object_name = bucket_and_object.split("/", 1)
            bucket_obj = storage_client.bucket(bucket_name)
            blob = bucket_obj.blob(object_name)

            # Check if file exists
            if not blob.exists():
                raise FileNotFoundError(
                    f"File not found: {cloud_path}\n\n"
                    f"Please verify the path and ensure the export task has completed successfully."
                )

            # Determine file format from extension
            file_ext = object_name.lower().split(".")[-1] if "." in object_name else ""

            LOGGER.warning(f"Detected file format: {file_ext}")

            # Download file content
            file_content = blob.download_as_bytes()

            # Parse based on file format
            if file_ext == "csv":
                # CSV file - read as DataFrame
                LOGGER.warning("Reading CSV file as DataFrame")
                df = pd.read_csv(io.BytesIO(file_content))
                LOGGER.warning(f"Successfully loaded CSV with {len(df)} rows")

            elif file_ext == "geojson":
                # GeoJSON file - read as GeoDataFrame
                LOGGER.warning("Reading GeoJSON file as GeoDataFrame")
                df = gpd.read_file(io.BytesIO(file_content))
                LOGGER.warning(f"Successfully loaded GeoJSON with {len(df)} rows")

            elif file_ext in ["kml", "kmz"]:
                # KML/KMZ file - read as GeoDataFrame
                LOGGER.warning(f"Reading {file_ext.upper()} file as GeoDataFrame")
                # For KMZ, we need to handle zip extraction
                if file_ext == "kmz":
                    import zipfile

                    with zipfile.ZipFile(io.BytesIO(file_content)) as z:
                        # Find the KML file inside
                        kml_files = [f for f in z.namelist() if f.endswith(".kml")]
                        if not kml_files:
                            raise ValueError("No KML file found in KMZ archive")
                        kml_content = z.read(kml_files[0])
                        df = gpd.read_file(io.BytesIO(kml_content), driver="KML")
                else:
                    df = gpd.read_file(io.BytesIO(file_content), driver="KML")
                LOGGER.warning(
                    f"Successfully loaded {file_ext.upper()} with {len(df)} rows"
                )

            elif file_ext == "shp":
                # Shapefile - read as GeoDataFrame
                # Note: Shapefile consists of multiple files, but we'll try to read the .shp
                LOGGER.warning("Reading Shapefile as GeoDataFrame")
                # For shapefiles, we might need to download all related files
                # But for simplicity, try to read if it's a single file
                # In practice, shapefiles exported from GEE might need special handling
                try:
                    df = gpd.read_file(io.BytesIO(file_content))
                except Exception as e:
                    raise ValueError(
                        f"Shapefile reading failed: {e}\n\n"
                        f"Note: Shapefiles exported from GEE consist of multiple files (.shp, .shx, .dbf, .prj). "
                        f"Consider downloading the entire folder from Cloud Storage or use GeoJSON format instead."
                    )
                LOGGER.warning(f"Successfully loaded Shapefile with {len(df)} rows")

            else:
                # Try to auto-detect format
                LOGGER.warning(
                    f"Unknown file extension '{file_ext}', attempting to auto-detect format"
                )
                try:
                    # Try GeoJSON first (most common for geospatial data)
                    df = gpd.read_file(io.BytesIO(file_content))
                    LOGGER.warning("Auto-detected as GeoJSON/GeoDataFrame")
                except Exception:
                    try:
                        # Try CSV
                        df = pd.read_csv(io.BytesIO(file_content))
                        LOGGER.warning("Auto-detected as CSV/DataFrame")
                    except Exception as e:
                        raise ValueError(
                            f"Unable to determine file format for '{self.cloud_path}'. "
                            f"Supported formats: CSV, GeoJSON, KML, KMZ, SHP. "
                            f"Error: {e}"
                        )

            # Ensure CRS is set for GeoDataFrames
            if isinstance(df, gpd.GeoDataFrame):
                if df.crs is None:
                    df.set_crs(epsg=4326, inplace=True)
                    LOGGER.warning("Set CRS to EPSG:4326 (WGS84)")

            # Remove RowID column if present (from GEE exports)
            if "<RowID>" in df.columns:
                df = df.drop(columns=["<RowID>"])

            LOGGER.warning(
                f"Successfully converted Cloud Storage file to table with {len(df)} rows"
            )

            return knext.Table.from_pandas(df)

        except Exception as e:
            LOGGER.error(f"Cloud Storage to Table conversion failed: {e}")
            raise
