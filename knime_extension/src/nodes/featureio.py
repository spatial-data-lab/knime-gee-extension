"""
GEE Feature Collection nodes for KNIME.
Feature collection I/O and processing: read, filter, clip, export.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_feature_collection_port_type,
    gee_image_port_type,
)

# Category for GEE Feature Collection I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="featureio",
    name="Feature Collection",
    description="Feature collection I/O and processing: read, filter, clip, export.",
    icon="icons/featureIO.png",
    after="authorization",
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

        collection_id = (self.collection_id or "").strip()
        if not collection_id:
            raise ValueError(
                "Collection ID cannot be empty. Please provide a valid GEE feature collection ID."
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
        if asset_type not in ("FEATURE_COLLECTION", "TABLE"):
            suggested = {
                "IMAGE": "GEE Image Reader",
                "IMAGE_COLLECTION": "GEE Image Collection Reader",
            }.get(asset_type, "the appropriate reader node")
            raise ValueError(
                f"Asset '{collection_id}' is not a Feature Collection (type: {asset_type}). "
                f"Please use {suggested}."
            )

        # Load feature collection
        feature_collection = ee.FeatureCollection(collection_id)

        try:
            LOGGER.warning(f"Loaded feature collection: {collection_id}")
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

    To get a list of available properties, use the "Feature Collection Info Extractor" node.

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
        "Filter operator",
        "The comparison operator to use for filtering.",
        default_value="Equals",
        enum=list(OPERATOR_CHOICES.keys()),
    )

    property_name = knext.StringParameter(
        "Property name",
        """Name of the property to filter by (e.g., 'ADM0_NAME'). To get a list of available properties, 
        use the "Feature Collection Info Extractor" node.""",
        default_value="",
    )

    property_value = knext.StringParameter(
        "Property value",
        "Value to filter by. For 'Is In List', use comma-separated values (e.g., 'China,Japan,India').",
        default_value="",
    )

    max_features = knext.IntParameter(
        "Maximum features",
        "Maximum number of features to return (-1 = no limit).",
        default_value=-1,
        min_value=-1,
        is_advanced=True,
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
# GEE Feature Collection to Point
############################################


@knext.node(
    name="Feature Collection to Point",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCtoPoint.png",
    id="fctopoint",
    after="valuefilter",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection (polygons, lines, or points) to convert to points.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection with point geometry (one centroid per feature).",
    port_type=gee_feature_collection_port_type,
)
class GEEFeatureCollectionToPoint:
    """Replace each feature's geometry with its centroid, one point per feature.

    **This node replaces each feature geometry with its centroid and is commonly used
    to convert polygons or lines into representative points for sampling or joins.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, fc_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        fc = fc_connection.feature_collection
        result = fc.map(lambda f: f.setGeometry(f.geometry().centroid()))
        LOGGER.warning(
            "Feature Collection to Point: geometry set to centroid per feature."
        )
        return knut.export_gee_feature_collection_connection(result, fc_connection)


############################################
# GEE Feature Collection Spatial Filter
############################################


@knext.node(
    name="Feature Collection Spatial Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "spatialFilter.png",
    id="spatialfilter",
    after="fctopoint",
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
    """Performs spatial filtering and clipping on Feature Collections.

    This node filters features using Spatial operator and Distance parameters and is
    commonly used to select features that intersect, contain, or lie within a distance of a boundary.

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
        "Spatial operator",
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
# Feature Collection Join (Spatial Join)
############################################


@knext.node(
    name="Feature Collection Join",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCJoin.png",
    id="fcjoin",
    after="spatialfilter",
)
@knext.input_port(
    name="Primary Feature Collection",
    description="The primary Feature Collection (e.g. blocks, polygons).",
    port_type=gee_feature_collection_port_type,
)
@knext.input_port(
    name="Secondary Feature Collection",
    description="The secondary Feature Collection to join (e.g. roads, points).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="Joined Feature Collection with matches according to join type.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionJoin:
    """Spatial join of two Feature Collections such as blocks within 1 km of highways or points in polygons.

    This node performs a spatial join between a primary and secondary Feature Collection using Spatial filter, Distance, and Join type parameters, and is commonly used to attach nearby features or count relationships.

    **Parameters:**
    - **Spatial filter:**
      - **withinDistance**: match secondary features within Distance of primary geometry.
      - **intersects**: match secondary features that intersect primary geometry.
    - **Distance (meters):** maximum search distance for withinDistance; ignored for intersects.
    - **Join type:**
      - **simple**: keep primary features that have at least one match.
      - **saveAll**: attach a list of all matching secondary features to each primary feature.
    - **Matches property name:** name of the property that stores matches for saveAll.
    - **Max error (meters):** spatial tolerance; larger values can speed up geometry operations.

    **Common Use Cases:**
    - Count points in polygons (e.g., schools per district).
    - Find polygons within a buffer of roads or rivers.
    - Attach nearby facilities to administrative units.

    **Performance Notes:**
    - Use **intersects** when boundaries overlap; use **withinDistance** only when proximity is required.
    - Smaller Distance and larger Max error generally improve performance.
    - Simplify or dissolve inputs upstream to reduce geometry complexity.

    **Usage Notes:**
    - For counts with saveAll, use **Feature Collection Calculator** with `size(matches)` to create a count property.
    - The output stays as a Feature Collection; convert to table only if the collection is small.

    """

    join_filter = knext.StringParameter(
        "Spatial filter",
        "Condition for matching: within distance (meters) or intersects.",
        default_value="withinDistance",
        enum=["withinDistance", "intersects"],
    )

    distance_meters = knext.DoubleParameter(
        "Distance (meters)",
        "Maximum distance for withinDistance filter. Ignored for intersects.",
        default_value=1000.0,
        min_value=0.0,
    ).rule(knext.OneOf(join_filter, ["withinDistance"]), knext.Effect.SHOW)

    join_type = knext.StringParameter(
        "Join type",
        "Simple: keep primary features that have at least one match. Save-all: add a property with list of matches.",
        default_value="simple",
        enum=["simple", "saveAll"],
    )

    matches_key = knext.StringParameter(
        "Matches property name",
        "Property name for the list of matching secondary features (save-all join only).",
        default_value="matches",
    ).rule(knext.OneOf(join_type, ["saveAll"]), knext.Effect.SHOW)

    max_error = knext.DoubleParameter(
        "Max error (meters)",
        "Spatial filter tolerance; larger values can speed up computation.",
        default_value=10.0,
        min_value=0.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        primary_fc_connection,
        secondary_fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        primary = primary_fc_connection.feature_collection
        secondary = secondary_fc_connection.feature_collection

        if self.join_filter == "withinDistance":
            the_filter = ee.Filter.withinDistance(
                leftField=".geo",
                rightField=".geo",
                distance=self.distance_meters,
                maxError=self.max_error,
            )
        else:
            the_filter = ee.Filter.intersects(
                leftField=".geo",
                rightField=".geo",
                maxError=self.max_error,
            )

        if self.join_type == "saveAll":
            join = ee.Join.saveAll(
                matchesKey=self.matches_key,
            )
            result = join.apply(primary, secondary, the_filter)
        else:
            join = ee.Join.simple()
            result = join.apply(primary, secondary, the_filter)

        LOGGER.warning(
            "Feature Collection Join applied: filter=%s, joinType=%s",
            self.join_filter,
            self.join_type,
        )
        return knut.export_gee_feature_collection_connection(
            result, primary_fc_connection
        )


############################################
# Feature Collection Calculator (expression → new property)
############################################


def _tokenize_fc_calc_expression(s):
    """Tokenize FC Calculator expression: numbers, identifiers, size(id), + - * / ( )."""
    tokens = []
    i = 0
    s = s.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] in "()+*-/":
            sym = {
                "(": "LPAREN",
                ")": "RPAREN",
                "+": "PLUS",
                "-": "MINUS",
                "*": "STAR",
                "/": "SLASH",
            }[s[i]]
            tokens.append((sym, None))
            i += 1
            continue
        if s[i].isdigit() or (s[i] == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            if s[j] == ".":
                j += 1
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            try:
                num = float(s[i:j])
                tokens.append(("NUMBER", num))
            except ValueError:
                raise ValueError(f"Invalid number: {s[i:j]!r}")
            i = j
            continue
        if s[i].isalpha() or s[i] == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            ident = s[i:j]
            tokens.append(("SIZE" if ident.lower() == "size" else "IDENT", ident))
            i = j
            continue
        raise ValueError(f"Unexpected character: {s[i]!r}")
    return tokens


def _parse_fc_calc_expression(expression):
    """Parse FC Calculator expression into AST. Raises ValueError on error."""
    tokens = _tokenize_fc_calc_expression(expression)
    if not tokens:
        raise ValueError("Expression is empty")
    pos = [0]

    def peek():
        if pos[0] >= len(tokens):
            return (None, None)
        return tokens[pos[0]]

    def consume():
        t = peek()
        pos[0] += 1
        return t

    def parse_expr():
        left = parse_term()
        while True:
            typ, _ = peek()
            if typ == "PLUS":
                consume()
                right = parse_term()
                left = ("BINOP", "+", left, right)
            elif typ == "MINUS":
                consume()
                right = parse_term()
                left = ("BINOP", "-", left, right)
            else:
                break
        return left

    def parse_term():
        left = parse_factor()
        while True:
            typ, _ = peek()
            if typ == "STAR":
                consume()
                right = parse_factor()
                left = ("BINOP", "*", left, right)
            elif typ == "SLASH":
                consume()
                right = parse_factor()
                left = ("BINOP", "/", left, right)
            else:
                break
        return left

    def parse_factor():
        typ, val = consume()
        if typ == "NUMBER":
            return ("NUMBER", val)
        if typ == "IDENT":
            return ("IDENT", val)
        if typ == "LPAREN":
            node = parse_expr()
            t, _ = consume()
            if t != "RPAREN":
                raise ValueError("Missing ')'")
            return node
        if typ == "SIZE":
            t2, _ = consume()
            if t2 != "LPAREN":
                raise ValueError("size() requires opening parenthesis")
            ident_typ, ident_val = consume()
            if ident_typ != "IDENT":
                raise ValueError("size(propertyName) requires a property name")
            t3, _ = consume()
            if t3 != "RPAREN":
                raise ValueError("Missing ')' after size(propertyName)")
            return ("SIZE", ident_val)
        raise ValueError(f"Unexpected token: {typ}")

    ast = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError("Unexpected tokens at end of expression")
    return ast


def _codegen_fc_calc(ast):
    """Generate Python expression string that builds ee object from feature f. Uses ee and f."""
    if ast[0] == "NUMBER":
        return f"ee.Number({ast[1]})"
    if ast[0] == "IDENT":
        name = ast[1]
        return f"ee.Number(f.get({repr(name)}))"
    if ast[0] == "SIZE":
        name = ast[1]
        return f"ee.List(f.get({repr(name)})).size()"
    if ast[0] == "BINOP":
        _, op, left, right = ast
        a = _codegen_fc_calc(left)
        b = _codegen_fc_calc(right)
        if op == "+":
            return f"({a}).add({b})"
        if op == "-":
            return f"({a}).subtract({b})"
        if op == "*":
            return f"({a}).multiply({b})"
        if op == "/":
            return f"({a}).divide({b})"
    raise ValueError(f"Unknown AST node: {ast[0]}")


@knext.node(
    name="Feature Collection Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCCalculator.png",
    id="fccalculator",
    after="fcjoin",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection to add a computed property to.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection with the new property added to each feature.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionCalculator:
    __doc__ = (
        "Add a computed property to each feature using an expression similar to Band Calculator for features.\n\n"
        "This node computes a new property using Expression and Output property name parameters and is commonly used for densities, ratios, or derived attributes.\n\n"
        "**Parameters:**\n"
        "- **Expression:** Formula using feature properties and helpers.\n"
        "- **Output property name:** Name of the new attribute.\n\n"
        "**Usage notes:**\n"
        "- Use Feature Collection Geometry Calculator to add area/length before computing densities.\n\n"
        + knut.FC_CALC_EXPRESSION_SYMBOLS_AND_EXAMPLES
    )

    expression = knext.StringParameter(
        "Expression",
        "Expression using feature property names. "
        + knut.FC_CALC_EXPRESSION_SYMBOLS
        + " Examples: pop10 / area_sq_mi, size(trees).",
        default_value="pop10 / area_sq_mi",
    )

    output_property_name = knext.StringParameter(
        "Output property name",
        "Name of the property to set with the computed value.",
        default_value="pop_density",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        expression = (self.expression or "").strip()
        output_name = (self.output_property_name or "calculated").strip()
        if not expression:
            raise ValueError("Expression is required.")
        if not output_name:
            raise ValueError("Output property name is required.")

        try:
            ast = _parse_fc_calc_expression(expression)
            expr_code = _codegen_fc_calc(ast)
        except ValueError as e:
            raise ValueError(f"Invalid expression: {e}") from e

        mapper_code = (
            "def _mapper(f):\n"
            "  f = ee.Feature(f)\n"
            f"  return f.set({repr(output_name)}, {expr_code})\n"
        )
        scope = {"ee": ee}
        exec(mapper_code, scope)
        _mapper = scope["_mapper"]

        fc = fc_connection.feature_collection
        result = fc.map(_mapper)

        LOGGER.warning(
            "Feature Collection Calculator: output property '%s'",
            output_name,
        )
        return knut.export_gee_feature_collection_connection(result, fc_connection)


############################################
# Feature Collection Geometry Calculator
############################################


class _GeometryAttributesOptions(knext.EnumParameterOptions):
    """Which geometry-derived attributes to add (multi-select)."""

    AREA = ("Area", "Add area (in selected unit)")
    LENGTH = ("Length", "Add length (in selected unit)")
    PERIMETER = ("Perimeter", "Add perimeter (in selected unit)")
    LATITUDE = ("Latitude", "Add centroid latitude (degrees WGS84)")
    LONGITUDE = ("Longitude", "Add centroid longitude (degrees WGS84)")

    @classmethod
    def get_default(cls):
        return [cls.AREA.name]


class _GeometryUnitOptions(knext.EnumParameterOptions):
    """Unit for area, length, and perimeter."""

    METER = ("Meter", "Area in m², length/perimeter in m")
    KILOMETER = ("Kilometer", "Area in km², length/perimeter in km")
    MILE = ("Mile", "Area in sq mi, length/perimeter in mi")

    @classmethod
    def get_default(cls):
        return cls.METER


@knext.node(
    name="Feature Collection Geometry Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCGeometry.png",
    id="fcgeometrycalculator",
    after="fccalculator",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection to add geometry attributes to.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection with added geometry attributes.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionGeometryCalculator:
    """Add geometry-derived attributes to each feature: area, length, perimeter, and/or centroid latitude/longitude.

    This node adds geometry attributes using Attributes to add and Unit parameters and is commonly used to compute area, length, or centroid coordinates for further analysis.

    **Parameters:**
    - **Attributes to add:** Area, length, perimeter, latitude, longitude.
    - **Unit:** Unit for area, length, and perimeter.

    **Usage notes:**
    - Latitude and longitude are derived from centroid coordinates.
    - **Attributes** (multi-select): Area, Length, Perimeter use the chosen **Unit** (meter, kilometer, mile).
    - **Latitude** and **Longitude** are always in degrees (WGS84), from the feature geometry centroid.

    - **Default property names:** ``area``, ``length``, ``perimeter``, ``latitude``, ``longitude``.
    Use with **Feature Collection Calculator** e.g. ``pop10 / area`` for density when unit is Mile (area in sq mi).
    """

    attributes = knext.EnumSetParameter(
        "Attributes to add",
        "Select one or more geometry-derived attributes. Lat/Lon are centroid coordinates in degrees.",
        default_value=_GeometryAttributesOptions.get_default(),
        enum=_GeometryAttributesOptions,
    )

    unit = knext.EnumParameter(
        "Unit",
        "Unit for area, length, and perimeter. Latitude and longitude are always in degrees.",
        default_value=_GeometryUnitOptions.get_default().name,
        enum=_GeometryUnitOptions,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        fc = fc_connection.feature_collection
        # EnumSetParameter returns a list of option names (e.g. ["AREA", "LENGTH"])
        selected = self.attributes or _GeometryAttributesOptions.get_default()
        if not isinstance(selected, list):
            selected = [selected] if selected else []
        unit_name = self.unit or _GeometryUnitOptions.get_default().name

        # Scale factors: GEE returns area in m², length/perimeter in m
        if unit_name == _GeometryUnitOptions.METER.name:
            area_scale = 1.0
            length_scale = 1.0
        elif unit_name == _GeometryUnitOptions.KILOMETER.name:
            area_scale = 1.0 / 1e6  # m² -> km²
            length_scale = 1.0 / 1000  # m -> km
        else:  # MILE
            area_scale = 1.0 / 2.589988e6  # m² -> sq mi
            length_scale = 1.0 / 1609.344  # m -> mi

        add_area = _GeometryAttributesOptions.AREA.name in selected
        add_length = _GeometryAttributesOptions.LENGTH.name in selected
        add_perimeter = _GeometryAttributesOptions.PERIMETER.name in selected
        add_lat = _GeometryAttributesOptions.LATITUDE.name in selected
        add_lon = _GeometryAttributesOptions.LONGITUDE.name in selected

        def mapper(f):
            f = ee.Feature(f)
            geom = f.geometry()
            out = f
            if add_area:
                out = out.set(
                    "area",
                    ee.Number(geom.area()).multiply(area_scale),
                )
            if add_length:
                out = out.set(
                    "length",
                    ee.Number(geom.length()).multiply(length_scale),
                )
            if add_perimeter:
                out = out.set(
                    "perimeter",
                    ee.Number(geom.perimeter()).multiply(length_scale),
                )
            if add_lat:
                # centroid().coordinates() is [lon, lat] in degrees
                lat = ee.Number(geom.centroid().coordinates().get(1))
                out = out.set("latitude", lat)
            if add_lon:
                lon = ee.Number(geom.centroid().coordinates().get(0))
                out = out.set("longitude", lon)
            return out

        result = fc.map(mapper)
        LOGGER.warning(
            "Feature Collection Geometry Calculator: attributes=%s, unit=%s",
            selected,
            unit_name,
        )
        return knut.export_gee_feature_collection_connection(result, fc_connection)


############################################
# Feature Collection to Image (Rasterize)
############################################


@knext.node(
    name="Feature Collection to Image",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCtoImage.png",
    id="fctoimage",
    after="fcgeometrycalculator",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection to rasterize (e.g. polygons to mask or property image).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection (rasterized mask or property values).",
    port_type=gee_image_port_type,
)
class FeatureCollectionToImage:
    """Rasterizes a Feature Collection to an Image using reduceToImage. Binary mask or property values per pixel.

    This node rasterizes features using Output mode, Property name, and Scale parameters and is commonly used to create masks or ID rasters from polygons.

    **Parameters:**
    - **Output:** binary mask or property image.
    - **Property name:** Numeric property to burn when using property mode.
    - **Scale:** Output pixel size in meters.
    - **Expand zeros to buffer area / Mask mode / Buffer distance:** Optional extent and mask controls.

    **Usage notes:**
    - Binary mode outputs band **mask**; property mode uses the property name.
    - **Binary mask:** Output 1 where a polygon intersects the pixel, 0 elsewhere (e.g. protected area mask).
    - **Property image:** Output the value of the given property for the polygon that covers the pixel (e.g. unique ID).

    Requires a reference geometry or scale for the output grid; use the bounds of the FC and the scale parameter.

    - Optional: expand the output extent by a buffer to keep explicit 0s in the buffer area.
    - Optional: invert or force mask values in binary mode.

    **Output band name:** Binary mode uses band **mask**; property mode uses the selected property name as the band name.
    Use band **mask** in Band Calculator or Band Merger (e.g. with Distance to FC output **distance**) without name clashes.
    """

    output_mode = knext.StringParameter(
        "Output",
        "Binary mask (1/0) or image from a numeric property (e.g. unique ID).",
        default_value="binary",
        enum=["binary", "property"],
    )

    property_name = knext.StringParameter(
        "Property name",
        "Numeric property to use as pixel value (e.g. WDPA_PID, zone_id). Used when Output is property.",
        default_value="",
    ).rule(knext.OneOf(output_mode, ["property"]), knext.Effect.SHOW)

    scale = knext.IntParameter(
        "Scale (meters)",
        "Pixel size of the output image.",
        default_value=500,
        min_value=1,
        max_value=10000,
    )

    expand_to_buffer = knext.BoolParameter(
        "Expand zeros to buffer area",
        "If enabled, expand the output extent by a buffer so areas outside the FC get explicit 0 values.",
        default_value=False,
    )

    mask_mode = knext.StringParameter(
        "Mask mode",
        "Binary mask polarity: inside=1/outside=0, inverted, or all ones. Only when expanding to buffer.",
        default_value="inside",
        enum=["inside", "inverted", "all_ones"],
    ).rule(knext.OneOf(expand_to_buffer, [True]), knext.Effect.SHOW)

    buffer_distance = knext.IntParameter(
        "Buffer distance (meters)",
        "Buffer distance in meters when expanding the output extent.",
        default_value=1000,
        min_value=1,
    ).rule(knext.OneOf(expand_to_buffer, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        fc = fc_connection.feature_collection

        if self.output_mode == "binary":
            # Add a constant property 'mask' = 1, then reduceToImage with first non-null
            fc_masked = fc.map(lambda f: ee.Feature(f).set("mask", 1))
            image = fc_masked.reduceToImage(
                properties=["mask"],
                reducer=ee.Reducer.first(),
            )
            image = image.unmask(0).rename("mask")
            if self.mask_mode == "inverted":
                image = image.eq(0).rename("mask")
            elif self.mask_mode == "all_ones":
                image = image.multiply(0).add(1).rename("mask")
        else:
            prop = (self.property_name or "").strip()
            if not prop:
                raise ValueError("Property name is required for property output.")
            image = fc.reduceToImage(
                properties=[prop],
                reducer=ee.Reducer.first(),
            )
            image = image.unmask(0)

        # Clip to FC bounds (optionally expand by buffer) and set scale for downstream use
        bounds = fc.geometry()
        if self.expand_to_buffer:
            clip_geom = bounds.buffer(self.buffer_distance)
        else:
            clip_geom = bounds

        image = image.clip(clip_geom).reproject(crs="EPSG:4326", scale=self.scale)

        LOGGER.warning(
            "Feature Collection rasterized: mode=%s, scale=%s m, maskMode=%s, expand=%s, buffer=%s m",
            self.output_mode,
            self.scale,
            self.mask_mode if self.output_mode == "binary" else "n/a",
            self.expand_to_buffer,
            self.buffer_distance if self.expand_to_buffer else 0,
        )
        return knut.export_gee_image_connection(image, fc_connection)


############################################
# Buffer Points
############################################


@knext.node(
    name="Buffer Features",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCBuffer.png",
    id="bufferpoints",
    after="fctoimage",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection of points, polygons, or lines to buffer.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection with buffered geometries (circles or square bounds).",
    port_type=gee_feature_collection_port_type,
)
class BufferFeatures:
    """Buffers each feature by a distance in meters. Output can be circular buffers or square bounds.

    This node buffers features using Buffer radius and Output square bounds parameters and is commonly used to create sample plots or distance-based zones.

    **Parameters:**
    - **Buffer radius:** Distance in meters.
    - **Output square bounds:** Use square bounds instead of circular buffers.

    **Usage notes:**
    - Works for points, lines, and polygons.

    Works for points (e.g. plot centers), polygons (e.g. expand protected area), or lines.
    Use with **Feature Collection Reducer** or **Local Region Reducer** for zonal statistics.
    """

    buffer_radius = knext.DoubleParameter(
        "Buffer radius (meters)",
        "Distance for buffer (radius of circle or half-side of square).",
        default_value=50.0,
        min_value=0.1,
    )

    use_bounds = knext.BoolParameter(
        "Output square bounds",
        "If enabled, output rectangular bounds of the buffer (square); otherwise circular polygon.",
        default_value=False,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        fc = fc_connection.feature_collection

        def buffer_feature(f):
            geom = ee.Feature(f).geometry()
            buffered = geom.buffer(self.buffer_radius)
            if self.use_bounds:
                buffered = buffered.bounds()
            return ee.Feature(buffered).copyProperties(f, exclude=["geometry"])

        result = fc.map(buffer_feature)
        LOGGER.warning(
            "Buffered features: radius=%s m, squareBounds=%s",
            self.buffer_radius,
            self.use_bounds,
        )
        return knut.export_gee_feature_collection_connection(result, fc_connection)


############################################
# Distance to Feature Collection
############################################


@knext.node(
    name="Distance to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "DistToFC.png",
    id="distancetofc",
    after="bufferpoints",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection whose boundary is used to compute distance (e.g. protected area).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection: each pixel value is distance (meters) to the FC boundary.",
    port_type=gee_image_port_type,
)
class DistanceToFeatureCollection:
    """Creates an image whose **distance** band is the distance in meters from each cell to the FC polygon boundary.

    This node builds a distance-to-boundary raster using Scale and Max distance parameters and is commonly used for proximity analysis, buffer modeling, and distance-based weighting.

    **Parameters:**
    - **Scale (meters):** Output pixel size for the distance image.
    - **Max distance (meters):** Maximum distance from the boundary to include; pixels farther away are masked.

    **Common Use Cases:**
    - Distance to protected area boundary for risk or accessibility modeling.
    - Distance to rivers/roads (after converting them to a Feature Collection).
    - Create distance decay inputs for suitability or cost surfaces.

    **Performance Notes:**
    - Smaller Scale and larger Max distance increase computation.
    - Clip or simplify the input Feature Collection to reduce geometry complexity.

    **Usage Notes:**
    - Output band name is **distance** in meters.
    - Output is masked beyond Max distance and reprojected to EPSG:4326 at the chosen Scale.
    - Combine with **Feature Collection to Image** mask in Band Merger so Band Calculator can use `distance` and `mask` together.
    """

    scale = knext.IntParameter(
        "Scale (meters)",
        "Pixel size of the output distance image.",
        default_value=500,
        min_value=1,
        max_value=5000,
    )

    max_distance = knext.IntParameter(
        "Max distance (meters)",
        "Only cells within this distance from the boundary have a value; beyond this, pixels are masked.",
        default_value=50000,
        min_value=100,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        fc = fc_connection.feature_collection
        bounds = fc.geometry()

        # Extent = polygon buffered by max_distance (only these cells).
        # FeatureCollection.distance(): each pixel = distance in meters to nearest polygon boundary.
        # Pixels beyond searchRadius stay masked; we do not fill them.
        dist_image = fc.distance(
            searchRadius=self.max_distance,
            maxError=50,
        )
        dist_image = dist_image.rename("distance").clip(
            bounds.buffer(self.max_distance)
        )
        dist_image = dist_image.reproject(crs="EPSG:4326", scale=self.scale)

        LOGGER.info(
            "Distance to Feature Collection: scale=%s m, maxDistance=%s m",
            self.scale,
            self.max_distance,
        )
        return knut.export_gee_image_connection(dist_image, fc_connection)


@knext.node(
    name="Feature Collection Info Extractor",
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
    name="Feature Collection Info Extractor Table",
    description="Table containing property names, types, geometry type, and number of features",
)
class GEEFeatureCollectionInfo:
    """Displays property information about a Google Earth Engine Feature Collection in table format.

    This node lists property names, types, geometry type, and feature count with no parameters and is commonly used to inspect Feature Collections before filtering or calculations.
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
            first_feature = feature_collection.first()
            prop_names = first_feature.propertyNames()
            geom_type = first_feature.geometry().type()

            def get_prop_type(prop_name):
                return ee.Algorithms.ObjectType(first_feature.get(prop_name))

            properties_info = ee.Dictionary.fromLists(
                prop_names, prop_names.map(get_prop_type)
            )
            result_info = ee.Dictionary(
                {"properties": properties_info, "geometry_type": geom_type}
            ).getInfo()

            props = result_info.get("properties") or {}
            geometry_type = result_info.get("geometry_type", "Unknown")

            try:
                n_features = feature_collection.size().getInfo()
            except Exception:
                n_features = "—"

            property_data = []
            for prop_name, prop_type in props.items():
                property_data.append(
                    {
                        "Property Name": prop_name,
                        "Property Type": str(prop_type).lower(),
                    }
                )
            property_data.append(
                {"Property Name": "geometry", "Property Type": geometry_type}
            )
            property_data.append(
                {
                    "Property Name": "number of features",
                    "Property Type": str(n_features),
                }
            )

            df = pd.DataFrame(property_data)
            return knext.Table.from_pandas(df)

        except Exception as e:
            LOGGER.error("Failed to get Feature Collection info: %s", e)
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

    This node converts a Feature Collection to a local table using Output format and is commonly used for small collections or previewing results.

    **Parameters:**
    - **Output format:** DataFrame or GeoDataFrame.

    **Usage notes:**
    - Interactive API has a payload limit; use Feature Collection Exporter for large data.
    This node converts a Google Earth Engine FeatureCollection to a KNIME table,
    allowing you to work with GEE vector data in standard tabular format.
    This node bridges GEE vector operations with KNIME's data processing capabilities,
    making it useful for exporting classification results, converting GEE vector analysis outputs,
    and processing GEE-generated point samples or administrative boundaries.

    **IMPORTANT - Data Size Limitations:**

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
        "Output format",
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

    This node uploads a GeoTable using Geometry column and Batch size parameters and is commonly used to bring local study areas or training samples into GEE.

    **Parameters:**
    - **Geometry column:** Column containing geometry.
    - **Batch size:** Features per batch for large uploads.

    **Usage notes:**
    - Large tables are batched automatically to avoid upload limits.
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
        "Geometry column",
        "Column containing geometry data",
        port_index=0,
    )

    batch_size = knext.IntParameter(
        "Batch size",
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

        # Drop extra geometry columns to avoid GeoDataFrame.to_file errors
        geometry_columns = list(shp.select_dtypes(include="geometry").columns)
        extra_geometry_columns = [
            col for col in geometry_columns if col != self.geo_col
        ]
        if extra_geometry_columns:
            LOGGER.warning(
                "Dropping additional geometry columns: "
                + ", ".join(extra_geometry_columns)
            )
            shp = shp.drop(columns=extra_geometry_columns)
            shp = shp.set_geometry(self.geo_col)

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

    This node exports a Feature Collection using Destination mode, Export format, Destination path, and Wait for completion parameters and is commonly used to download large vector results.

    **Parameters:**
    - **Destination mode:** Drive or Cloud Storage.
    - **Export format:** CSV, GeoJSON, KML, KMZ, or SHP.
    - **Destination path:** Drive folder or Cloud Storage path.
    - **Wait for completion / Max wait seconds:** Optional blocking export.

    **Usage notes:**
    - Service accounts require Cloud Storage; Drive needs interactive auth with Drive scope.
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
        label="Destination mode",
        description="Select export destination. Drive requires interactive authentication with Drive scope.",
        default_value=DestinationModeOptions.get_default().name,
        enum=DestinationModeOptions,
    )

    export_format = knext.StringParameter(
        "Export format",
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
        "Destination path",
        "Destination path (Drive: 'DriveFolder/file'; Cloud Storage: 'bucket/path/file').",
        default_value="GEEexport/feature_collection_export",
    )

    wait_for_completion = knext.BoolParameter(
        "Wait for completion",
        "If enabled, the node waits until the export task completes.",
        default_value=False,
    )

    max_wait_seconds = knext.IntParameter(
        "Max wait seconds",
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

    This node reads a Cloud Storage file using Cloud Storage path and is commonly used to bring exported CSV or vector files back into KNIME.

    **Parameters:**
    - **Cloud Storage path:** Path to the file in a bucket.

    **Usage notes:**
    - File extension determines parser; supported: CSV, GeoJSON, KML/KMZ, SHP.
    Provide the file as a single Cloud Storage path such as ``bucket/path/file.ext`` or
    ``gs://bucket/path/file.ext``. The bucket must already exist and the filename must include
    the desired extension so the loader can detect CSV, GeoJSON, KML/KMZ, or SHP formats.
    """

    cloud_path = knext.StringParameter(
        "Cloud Storage path",
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
