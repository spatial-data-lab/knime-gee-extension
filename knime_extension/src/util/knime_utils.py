# this is a placeholder file for the knime_utils.py file
import logging
from typing import Callable
from typing import List
import knime_extension as knext


# get feature collection column names
def get_fc_columns(fc):
    """Get the column names of a feature collection.
    Args:
        fc (ee.FeatureCollection): The feature collection to get the column names from.
    Returns:
        list: The list of column names.
    """
    return fc.first().propertyNames().getInfo()


def negate(function):
    """
    Negates the incoming function e.g. negate(is_numeric) can be used in a column parameter to allow the user
    to select from all none numeric columns.
    @return: the negated input function e.g. if the input function returns true this function returns false
    """

    def new_function(*args, **kwargs):
        return not function(*args, **kwargs)

    return new_function


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def boolean_and(*functions):
    """
    Return True if all of the given functions return True
    @return: True if all of the functions return True
    """

    def new_function(*args, **kwargs):
        return all(f(*args, **kwargs) for f in functions)

    return new_function


def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return (
        column.ktype == knext.double()
        or column.ktype == knext.int32()
        or column.ktype == knext.int64()
    )


def is_int(column: knext.Column) -> bool:
    """
    Checks if column is integer (int32 only).
    @return: True if Column is integer
    """
    return column.ktype == knext.int32()


def is_long(column: knext.Column) -> bool:
    """
    Checks if column is long (e.g. int32 or int64).
    @return: True if Column is long
    """
    return column.ktype in [
        knext.int32(),
        knext.int64(),
    ]


def is_string(column: knext.Column) -> bool:
    """
    Checks if column is string
    @return: True if Column is string
    """
    return column.ktype == knext.string()


def is_boolean(column: knext.Column) -> bool:
    """
    Checks if column is boolean
    @return: True if Column is boolean
    """
    return column.ktype == knext.boolean()


def is_numeric_or_string(column: knext.Column) -> bool:
    """
    Checks if column is numeric or string
    @return: True if Column is numeric or string
    """
    return boolean_or(is_numeric, is_string)(column)


def is_int_or_string(column: knext.Column) -> bool:
    """
    Checks if column is int or string
    @return: True if Column is numeric or string
    """
    return column.ktype in [
        knext.int32(),
        knext.int64(),
        knext.string(),
    ]


def is_binary(column: knext.Column) -> bool:
    """
    Checks if column is binary
    @return: True if Column is binary
    """
    return column.ktype == knext.blob()


def is_date(column: knext.Column) -> bool:
    """
    Checks if column is compatible to the GeoValue interface and thus returns true for all geospatial types such as:
    GeoPointCell, GeoLineCell, GeoPolygonCell, GeoMultiPointCell, GeoMultiLineCell, GeoMultiPolygonCell, ...
    @return: True if Column Type is GeoValue compatible
    """
    return __is_type_x(column, "org.knime.core.data.v2.time.LocalDateValueFactory")


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type whereas type can be :
    GeoPointCell, GeoLineCell, GeoPolygonCell, GeoMultiPointCell, GeoMultiLineCell, GeoMultiPolygonCell, ...
    @return: True if Column Type is a GeoLogical Point
    """
    return (
        isinstance(column.ktype, knext.LogicalType)
        and type in column.ktype.logical_type
    )


__CELL_TYPE_GEO = "org.knime.geospatial.core.data.cell.Geo"


def is_geo(column: knext.Column) -> bool:
    """
    Checks if column is compatible to the GeoValue interface and thus returns true for all geospatial types such as:
    GeoPointCell, GeoLineCell, GeoPolygonCell, GeoMultiPointCell, GeoMultiLineCell, GeoMultiPolygonCell, ...
    @return: True if Column Type is GeoValue compatible
    """
    return __is_type_x(column, __CELL_TYPE_GEO)


############################################
# General helper
############################################


def column_exists_or_preset(
    context: knext.ConfigurationContext,
    column: str,
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = None,
    none_msg: str = "No compatible column found in input table",
) -> str:
    """
    Checks that the given column is not None and exists in the given schema. If none is selected it returns the
    first column that is compatible with the provided function. If none is compatible it throws an exception.
    """
    if column is None:
        for c in schema:
            if func(c):
                return c.name
        raise knext.InvalidParametersError(none_msg)
    __check_col_and_type(column, schema, func)
    return column


__DEF_PLEASE_SELECT_COLUMN = "Please select a column"


def column_exists(
    column: str,
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = None,
    none_msg: str = __DEF_PLEASE_SELECT_COLUMN,
) -> None:
    """
    Checks that the given column is not None and exists in the given schema otherwise it throws an exception
    """
    if column is None:
        raise knext.InvalidParametersError(none_msg)
    __check_col_and_type(column, schema, func)


def __check_col_and_type(
    column: str,
    schema: knext.Schema,
    check_type: Callable[[knext.Column], bool] = None,
) -> None:
    """
    Checks that the given column exists in the given schema and that it matches the given type_check function.
    """
    # Check that the column exists in the schema and that it has a compatible type
    try:
        existing_column = schema[column]
        if check_type is not None and not check_type(existing_column):
            raise knext.InvalidParametersError(
                f"Column '{str(column)}' has incompatible data type"
            )
    except IndexError:
        raise knext.InvalidParametersError(
            f"Column '{str(column)}' not available in input table"
        )


def columns_exist(
    columns: List[str],
    schema: knext.Schema,
    func: Callable[[knext.Column], bool] = lambda c: True,
    none_msg: str = __DEF_PLEASE_SELECT_COLUMN,
) -> None:
    """
    Checks that the given columns are not None and exist in the given schema otherwise it throws an exception
    """
    for col in columns:
        column_exists(col, schema, func, none_msg)


def fail_if_column_exists(
    column_name: str, input_schema: knext.Schema, msg: str = None
):
    """Checks that the given column name does not exists in the input schema.
    Can be used to check that a column is not accidentally overwritten."""
    if column_name in input_schema.column_names:
        if msg is None:
            msg = f"Column '{column_name}' exists"
        raise knext.InvalidParametersError(msg)


def get_unique_column_name(column_name: str, input_schema: knext.Schema) -> str:
    """Checks if the column name exists in the given schema and if so appends a number to it to make it unique.
    The unique name if returned or the original if it was already unique."""
    return get_unique_name(column_name, input_schema.column_names)


def get_unique_name(column_name: str, existing_col_names) -> str:
    """Checks if the column name exists in the given schema and if so appends a number to it to make it unique.
    The unique name if returned or the original if it was already unique."""
    if column_name is None:
        raise knext.InvalidParametersError("Column name must not be None")
    uniquifier = 1
    result = column_name
    while result in existing_col_names:
        result = column_name + f"(#{uniquifier})"
        uniquifier += 1
    return result


def check_canceled(exec_context: knext.ExecutionContext) -> None:
    """
    Checks if the user has canceled the execution and if so throws a RuntimeException
    """
    if exec_context.is_canceled():
        raise RuntimeError("Execution canceled")


def ensure_file_extension(file_name: str, file_extension: str) -> str:
    """
    Checks if the given file_name ends with the given file_extension and if not appends it to the returned file_name.
    """
    if not file_name:
        raise knext.InvalidParametersError("Please enter a valid file name")
    if file_name.lower().endswith(file_extension):
        return file_name
    return file_name + file_extension


############################################
# GeoPandas helper
############################################
# def load_geo_data_frame(
#     input_table: knext.Table,
#     column: knext.Column,
#     exec_context: knext.ExecutionContext = None,
#     load_msg: str = "Loading Geo data frame...",
#     done_msg: str = "Geo data frame loaded. Start computation...",
# ) -> gp.GeoDataFrame:
#     """Creates a GeoDataFrame from the given input table using the provided column as geo column."""
#     if exec_context:
#         exec_context.set_progress(0.0, load_msg)
#     import geopandas as gp

#     gdf = gp.GeoDataFrame(input_table.to_pandas(), geometry=column)
#     if exec_context:
#         exec_context.set_progress(0.1, done_msg)
#     return gdf


# def to_table(
#     gdf: gp.GeoDataFrame,
#     exec_context: knext.ExecutionContext = None,
#     done_msg: str = "Computation done",
# ) -> knext.Table:
#     """Returns a KNIME table representing the given GeoDataFrame."""
#     if exec_context:
#         exec_context.set_progress(1.0, done_msg)
#     return knext.Table.from_pandas(gdf)


############################################
# GEE helper
############################################


def create_gee_connection_object(gee_object, credentials, project_id):
    """
    Create a standardized GEE connection object with embedded GEE object.

    This function creates a GoogleEarthEngineConnectionObject that contains the GEE object,
    credentials, and project ID. This ensures all connected nodes run in the same Python process
    and share the GEE initialization state.

    Args:
        gee_object: The GEE object (ee.Image, ee.FeatureCollection, etc.)
        credentials: GEE credentials object (optional)
        project_id: GEE project ID string (optional)

    Returns:
        GoogleEarthEngineConnectionObject: Connection object with embedded GEE object
    """
    from util.common import (
        GoogleEarthEngineConnectionObject,
        GoogleEarthEngineObjectSpec,
    )

    # Create a new spec for the connection
    spec = GoogleEarthEngineObjectSpec(project_id)

    # Create connection object with embedded GEE object
    connection_object = GoogleEarthEngineConnectionObject(
        spec=spec, credentials=credentials, gee_object=gee_object
    )

    return connection_object


def export_gee_connection(gee_object, existing_connection):
    """
    Export a GEE object as a connection object using credentials and project ID from an existing connection.

    This function creates a GoogleEarthEngineConnectionObject that contains the GEE object,
    using the credentials and project ID from an existing connection. This ensures all connected
    nodes run in the same Python process and share the GEE initialization state.

    Args:
        gee_object: The GEE object (ee.Image, ee.FeatureCollection, etc.)
        existing_connection: Existing GoogleEarthEngineConnectionObject to get credentials and project_id from

    Returns:
        GoogleEarthEngineConnectionObject: Connection object with embedded GEE object
    """
    # Get credentials and project ID from existing connection
    credentials = existing_connection.credentials
    project_id = existing_connection.spec.project_id

    # Use the existing function to create the new connection object
    return create_gee_connection_object(gee_object, credentials, project_id)


def export_gee_image_connection(image, existing_connection):
    """
    Export a GEE Image as a specialized Image connection object.

    Args:
        image: The GEE Image object
        existing_connection: Existing connection object to get credentials and project_id from

    Returns:
        GEEImageConnectionObject: Specialized Image connection object
    """
    from util.common import (
        GEEImageConnectionObject,
        GEEImageObjectSpec,
    )

    credentials = existing_connection.credentials
    project_id = existing_connection.spec.project_id
    spec = GEEImageObjectSpec(project_id)

    return GEEImageConnectionObject(spec=spec, credentials=credentials, image=image)


def export_gee_feature_collection_connection(feature_collection, existing_connection):
    """
    Export a GEE FeatureCollection as a specialized FeatureCollection connection object.

    Args:
        feature_collection: The GEE FeatureCollection object
        existing_connection: Existing connection object to get credentials and project_id from

    Returns:
        GEEFeatureCollectionConnectionObject: Specialized FeatureCollection connection object
    """
    from util.common import (
        GEEFeatureCollectionConnectionObject,
        GEEFeatureCollectionObjectSpec,
    )

    credentials = existing_connection.credentials
    project_id = existing_connection.spec.project_id
    spec = GEEFeatureCollectionObjectSpec(project_id)

    return GEEFeatureCollectionConnectionObject(
        spec=spec, credentials=credentials, feature_collection=feature_collection
    )


def export_gee_classifier_connection(
    classifier,
    existing_connection,
    training_data=None,
    label_property=None,
    reverse_mapping=None,
    input_properties=None,
):
    """
    Export a GEE Classifier as a specialized Classifier connection object.

    Args:
        classifier: The trained GEE Classifier object
        existing_connection: Existing connection object to get credentials and project_id from
        training_data: Training data FeatureCollection (already remapped)
        label_property: Label property name used during training
        reverse_mapping: Reverse mapping for class values
        input_properties: Input properties (bands/features) used during training

    Returns:
        GEEClassifierConnectionObject: Specialized Classifier connection object
    """
    from util.common import (
        GEEClassifierConnectionObject,
        GEEClassifierObjectSpec,
    )

    credentials = existing_connection.credentials
    project_id = existing_connection.spec.project_id
    spec = GEEClassifierObjectSpec(project_id)

    return GEEClassifierConnectionObject(
        spec=spec,
        credentials=credentials,
        classifier=classifier,
        training_data=training_data,
        label_property=label_property,
        reverse_mapping=reverse_mapping,
        input_properties=input_properties,
    )


def export_gee_image_collection_connection(image_collection, existing_connection):
    """
    Export a GEE ImageCollection as a specialized ImageCollection connection object.

    Args:
        image_collection: The GEE ImageCollection object
        existing_connection: Existing connection object to get credentials and project_id from

    Returns:
        GEEImageCollectionConnectionObject: Specialized ImageCollection connection object
    """
    from util.common import (
        GEEImageCollectionConnectionObject,
        GEEImageCollectionObjectSpec,
    )

    credentials = existing_connection.credentials
    project_id = existing_connection.spec.project_id
    spec = GEEImageCollectionObjectSpec(project_id)

    return GEEImageCollectionConnectionObject(
        spec=spec, credentials=credentials, image_collection=image_collection
    )


def batch_process_feature_collection_to_table(
    feature_collection,
    file_format="DataFrame",
    batch_size=500,
    logger=None,
    exec_context=None,
    total_size=None,
):
    """
    Convert a Feature Collection to table using batch processing to handle GEE size limits.

    Uses ee.FeatureCollection.toList() + getInfo() instead of geemap.ee_to_df()
    to have better control over payload size and avoid payload limit errors.

    Args:
        feature_collection: ee.FeatureCollection to convert
        file_format: "DataFrame" or "GeoDataFrame" (default: "DataFrame")
        batch_size: Number of features per batch (default: 500)
        logger: Optional logger instance for progress messages
        exec_context: Optional ExecutionContext for progress updates
        total_size: Optional total size (if known) for progress calculation

    Returns:
        pandas.DataFrame: Converted table with all features
    """
    import ee
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import shape

    if logger is None:
        logger = logging.getLogger(__name__)

    # Try to get total size for progress display (optional)
    if total_size is None:
        try:
            total_size = feature_collection.size().getInfo()
            logger.warning(f"Feature Collection size: {total_size} features")
        except Exception as size_error:
            # Size check failed - that's OK, we'll use approximate progress
            logger.warning(
                f"Could not get collection size ({size_error}), will use approximate progress"
            )
            total_size = None

    result_dfs = []

    # Use user's batch_size (no automatic adjustment)
    current_batch_size = batch_size
    processed_count = 0
    batch_num = 0
    last_index = None

    while True:
        # Update progress
        if exec_context is not None:
            if total_size is not None:
                # Precise progress when we know total size
                progress = 0.1 + (processed_count / total_size) * 0.7
                exec_context.set_progress(
                    progress,
                    f"Processing batch {batch_num + 1}... ({processed_count}/{total_size} features)",
                )
            else:
                # Approximate progress when we don't know total size
                progress = min(0.1 + (batch_num * 0.01), 0.8)
                exec_context.set_progress(
                    progress,
                    f"Processing batch {batch_num + 1}... ({processed_count} features so far)",
                )

        try:
            # Get a batch using limit() - no size check to avoid payload issues
            if last_index is None:
                batch_fc = feature_collection.limit(current_batch_size)
            else:
                batch_fc = feature_collection.filter(
                    ee.Filter.gt("system:index", last_index)
                ).limit(current_batch_size)

            # Use toList() + getInfo() instead of geemap.ee_to_df()
            # This gives us more control and can handle smaller batches
            batch_list = batch_fc.toList(current_batch_size)
            batch_features = batch_list.getInfo()

            if not batch_features or len(batch_features) == 0:
                break

            # Manually convert features to DataFrame
            rows = []
            for feature in batch_features:
                props = feature.get("properties", {})
                geom = feature.get("geometry")

                row = props.copy()

                # Add geometry if needed for GeoDataFrame
                if file_format == "GeoDataFrame" and geom:
                    try:
                        row["geometry"] = shape(geom)
                    except Exception:
                        # If geometry conversion fails, skip geometry
                        pass

                rows.append(row)

            if len(rows) == 0:
                break

            # Create DataFrame
            if file_format == "GeoDataFrame":
                batch_df = gpd.GeoDataFrame(rows, crs="EPSG:4326")
            else:
                batch_df = pd.DataFrame(rows)
                # Remove geometry column if present (shouldn't be, but just in case)
                if "geometry" in batch_df.columns:
                    batch_df = batch_df.drop(columns=["geometry"])

            if len(batch_df) == 0:
                break

            # Get the last system:index for next iteration
            if "system:index" in batch_df.columns:
                batch_df_sorted = batch_df.sort_values("system:index")
                last_index = batch_df_sorted["system:index"].iloc[-1]
            else:
                # No system:index - can't continue efficiently
                logger.warning("No system:index found, stopping after this batch")
                result_dfs.append(batch_df)
                processed_count += len(batch_df)
                break

            result_dfs.append(batch_df)
            processed_count += len(batch_df)
            batch_num += 1

            # If we got fewer features than batch_size, we're done
            if len(batch_df) < current_batch_size:
                break

        except Exception as e:
            error_msg = str(e).lower()
            if "payload" in error_msg or "limit" in error_msg:
                # Don't adjust batch size - just raise error with helpful message
                if result_dfs:
                    # We have some results - return them and warn
                    logger.warning(
                        f"Payload limit error at batch {batch_num + 1}. "
                        f"Returning {len(result_dfs)} batches processed so far."
                    )
                    logger.warning(
                        f"Current batch size ({current_batch_size}) is too large. "
                        f"Please reduce the batch size parameter and try again."
                    )
                    break
                else:
                    # No results yet - provide detailed error message
                    if current_batch_size == 1:
                        # Single feature exceeds limit - this is a data issue
                        error_detail = (
                            f"Even a single feature exceeds the payload limit (10MB). "
                            f"This usually means:\n"
                            f"1. Features contain too many attributes or very large attribute values\n"
                            f"2. Features have very complex geometries\n"
                            f"3. Features contain embedded data that exceeds limits\n\n"
                            f"Suggestions:\n"
                            f"- Use 'Feature Collection Filter' to select only needed properties\n"
                            f"- Simplify geometries if possible\n"
                            f"- Consider using GEE Export functionality instead of direct conversion\n"
                            f"Original error: {e}"
                        )
                        raise ValueError(error_detail)
                    else:
                        # Batch size too large
                        error_detail = (
                            f"Batch size {current_batch_size} exceeds payload limit (10MB). "
                            f"Please reduce the batch size parameter to a smaller value (e.g., {max(1, current_batch_size // 4)}) "
                            f"and try again.\n"
                            f"Original error: {e}"
                        )
                        raise ValueError(error_detail)
            else:
                # Other errors - raise immediately
                raise

    # Set final progress
    if exec_context is not None:
        exec_context.set_progress(0.8, "Combining results...")

    # Combine all batches
    if result_dfs:
        df = pd.concat(result_dfs, ignore_index=True)
        logger.warning(f"Processed {len(df)} features in {len(result_dfs)} batches")
        return df
    else:
        raise ValueError("No batches were successfully processed")


def _process_batch(
    feature_collection,
    offset,
    current_batch_size,
    file_format,
    result_dfs,
    logger,
    batch_num,
    total_batches,
):
    """Helper function to process a single batch"""
    import ee
    import geemap

    # Get batch using toList - this returns ee.List
    batch_list = feature_collection.toList(current_batch_size, offset)

    # Convert ee.List to ee.FeatureCollection for geemap functions
    batch_fc = ee.FeatureCollection(batch_list)

    # Convert batch
    try:
        if file_format == "DataFrame":
            batch_df = geemap.ee_to_df(batch_fc)
        else:  # GeoDataFrame
            batch_df = geemap.ee_to_gdf(batch_fc)

        result_dfs.append(batch_df)
        batch_display = (
            f"{batch_num + 1}/{total_batches}" if total_batches else str(batch_num + 1)
        )
        logger.warning(f"Processed batch {batch_display} ({len(batch_df)} features)")
    except Exception as e:
        logger.warning(
            f"Batch {batch_num + 1} failed: {e}, trying smaller batch size..."
        )
        # Try with smaller batches for this range
        smaller_batch = max(current_batch_size // 2, 50)
        sub_result_dfs = []

        for j in range(offset, offset + current_batch_size, smaller_batch):
            sub_batch_size = min(smaller_batch, offset + current_batch_size - j)
            try:
                sub_batch_list = feature_collection.toList(sub_batch_size, j)
                sub_batch_fc = ee.FeatureCollection(sub_batch_list)
                if file_format == "DataFrame":
                    sub_df = geemap.ee_to_df(sub_batch_fc)
                else:
                    sub_df = geemap.ee_to_gdf(sub_batch_fc)
                sub_result_dfs.append(sub_df)
            except Exception as sub_error:
                logger.warning(
                    f"Sub-batch starting at {j} failed: {sub_error}, trying single features..."
                )
                # Try one feature at a time for this range
                for k in range(j, min(j + sub_batch_size, offset + current_batch_size)):
                    try:
                        single_list = feature_collection.toList(1, k)
                        single_fc = ee.FeatureCollection(single_list)
                        if file_format == "DataFrame":
                            single_df = geemap.ee_to_df(single_fc)
                        else:
                            single_df = geemap.ee_to_gdf(single_fc)
                        sub_result_dfs.append(single_df)
                    except Exception as single_error:
                        logger.error(
                            f"Failed to process single feature at index {k}: {single_error}"
                        )
                        # Skip this feature and continue
                        continue

        result_dfs.extend(sub_result_dfs)


def batch_process_geodataframe_to_feature_collection(
    gdf,
    batch_size=500,
    logger=None,
    exec_context=None,
):
    """
    Convert a GeoDataFrame to Feature Collection using batch processing to handle GEE upload limits.

    This function processes GeoDataFrames in batches to avoid GEE's upload size limits.
    It automatically handles errors by reducing batch size and provides detailed logging.

    Args:
        gdf: geopandas.GeoDataFrame to convert
        batch_size: Number of features per batch (default: 500)
        logger: Optional logger instance for progress messages
        exec_context: Optional ExecutionContext for progress updates

    Returns:
        ee.FeatureCollection: Combined Feature Collection with all features
    """
    import ee
    import geemap

    if logger is None:
        logger = logging.getLogger(__name__)

    total_size = len(gdf)
    num_batches = (total_size + batch_size - 1) // batch_size

    # Process in batches and combine Feature Collections
    feature_collections = []

    for i in range(num_batches):
        # Update progress
        if exec_context is not None:
            progress = 0.1 + (i / num_batches) * 0.7
            exec_context.set_progress(
                progress, f"Processing batch {i + 1}/{num_batches}"
            )

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)

        # Get batch of features
        batch_gdf = gdf.iloc[start_idx:end_idx].copy()

        try:
            # Convert batch to Feature Collection
            batch_fc = geemap.gdf_to_ee(batch_gdf)
            feature_collections.append(batch_fc)
            # logger.warning(
            #     f"Processed batch {i + 1}/{num_batches} ({len(batch_gdf)} features)"
            # )
        except Exception as e:
            # logger.warning(f"Batch {i + 1} failed: {e}, trying smaller batch size...")
            # Try with smaller batches for this range
            smaller_batch = max(len(batch_gdf) // 2, 50)
            sub_feature_collections = []

            for j in range(start_idx, end_idx, smaller_batch):
                sub_end_idx = min(j + smaller_batch, end_idx)
                sub_batch_gdf = gdf.iloc[j:sub_end_idx].copy()

                try:
                    sub_batch_fc = geemap.gdf_to_ee(sub_batch_gdf)
                    sub_feature_collections.append(sub_batch_fc)
                except Exception as sub_error:
                    # logger.warning(
                    #     f"Sub-batch starting at {j} failed: {sub_error}, trying single features..."
                    # )
                    # Try one feature at a time for this range
                    for k in range(j, sub_end_idx):
                        try:
                            single_gdf = gdf.iloc[k : k + 1].copy()
                            single_fc = geemap.gdf_to_ee(single_gdf)
                            sub_feature_collections.append(single_fc)
                        except Exception as single_error:
                            # logger.error(
                            #     f"Failed to process single feature at index {k}: {single_error}"
                            # )
                            # Skip this feature and continue
                            continue

            feature_collections.extend(sub_feature_collections)

    # Set final progress
    if exec_context is not None:
        exec_context.set_progress(0.8, "Combining Feature Collections...")

    # Combine all Feature Collections into one
    if feature_collections:
        # Flatten all feature collections into a single list
        all_features = []
        for fc in feature_collections:
            # Get features from each collection and add to list
            features_list = fc.toList(fc.size())
            all_features.append(features_list)

        # Combine all lists into one
        combined_list = all_features[0]
        for lst in all_features[1:]:
            combined_list = combined_list.cat(lst)

        # Create final Feature Collection from combined list
        combined_fc = ee.FeatureCollection(combined_list)

        return combined_fc
    else:
        raise ValueError("No batches were successfully processed")


def batch_process_gee_operation(
    operation_func, total_items, batch_size=500, logger=None, **operation_kwargs
):
    """
    Generic batch processing function for GEE operations that may exceed size limits.

    This function processes items in batches, calling operation_func for each batch,
    and combines the results. Useful for operations that need to avoid GEE size limits.

    Args:
        operation_func: Function that processes a batch and returns a result (e.g., DataFrame)
            Signature: operation_func(batch_data, batch_index, **operation_kwargs) -> result
        total_items: Total number of items to process
        batch_size: Number of items per batch (default: 500)
        logger: Optional logger instance for progress messages
        **operation_kwargs: Additional keyword arguments to pass to operation_func

    Returns:
        Combined result from all batches (typically a pandas.DataFrame)
    """
    import pandas as pd

    if logger is None:
        logger = logging.getLogger(__name__)

    num_batches = (total_items + batch_size - 1) // batch_size
    results = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_items)

        try:
            batch_result = operation_func(
                start_idx, end_idx, batch_index=i, **operation_kwargs
            )
            results.append(batch_result)
            # logger.warning(
            #     f"Processed batch {i + 1}/{num_batches} (items {start_idx}-{end_idx-1})"
            # )
        except Exception as e:
            # logger.warning(f"Batch {i + 1} failed: {e}, trying smaller batch size...")
            # Try with smaller batches for this range
            smaller_batch = max((end_idx - start_idx) // 2, 50)
            sub_results = []

            for j in range(start_idx, end_idx, smaller_batch):
                sub_end_idx = min(j + smaller_batch, end_idx)
                try:
                    sub_result = operation_func(
                        j, sub_end_idx, batch_index=i, **operation_kwargs
                    )
                    sub_results.append(sub_result)
                except Exception as sub_error:
                    # logger.error(
                    #     f"Sub-batch starting at {j} failed: {sub_error}, skipping..."
                    # )
                    # Skip this sub-batch and continue
                    continue

            results.extend(sub_results)

    # Combine all results
    if results:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results[0], list):
            # Flatten list of lists
            return [item for sublist in results for item in sublist]
        else:
            # Return as-is if not DataFrame or list
            return results
    else:
        raise ValueError("No batches were successfully processed")
