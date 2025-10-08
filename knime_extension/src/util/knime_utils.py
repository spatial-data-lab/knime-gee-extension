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
