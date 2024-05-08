# this is a placeholder file for the knime_utils.py file

# get feature collection column names
def get_fc_columns(fc):
    """Get the column names of a feature collection.
    Args:
        fc (ee.FeatureCollection): The feature collection to get the column names from.
    Returns:
        list: The list of column names.
    """
    return fc.first().propertyNames().getInfo()