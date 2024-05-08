import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="io",
    name="IO",
    description="Nodes that read and write from and to Google Earth Engine",
    # starting at the root folder of the extension_module parameter in the knime.yml file
    icon="icons/icon/IOCategory.png",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/io/"


############################################
#  Search Data From GEE data catalog Node
############################################
@knext.node(
    name="Search Data From GEE Data Catalog",
    node_type=knext.NodeType.SOURCE,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    category=__category,
    after="",
)
@knext.output_table(
    name="search result",
    description="Retrieved search results from GEE data catalog",
)
class SearchDataFromGEEDataCatalogNode:
    """This node searches data from Google Earth Engine (GEE) data catalog.
    Google Earth Engine (GEE) is a cloud-based platform for planetary-scale environmental data analysis.
    Please refer to [GEE](https://earthengine.google.com/) for more details.
    """

    search_keyword = knext.StringParameter(
        label="Search Keyword",
        description="The keyword to search from GEE data catalog.",
        default_value="SRTMGL1_003",
    )

    # max_items = knext.IntParameter(
    #     label="Max Items",
    #     description="The max items to search from GEE data catalog.",
    #     default_value=100,
    # )

    source = knext.StringParameter(
        label="Source",
        description="The source to search from GEE data catalog.",
        default_value="ee",
        enum=["ee", "community", "all"],
    )

    if_regex = knext.BoolParameter(
        label="Use Regular Expression",
        description="The flag to use regular expression to search from GEE data catalog.",
        default_value=False,
    )

    def configure(self, configure_context):
        # TODO Create combined schema
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee
        import pandas as pd
        from geemap import common as cm

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        # search_keyword = self.search_keyword
        # max_items = self.max_items

        search_result = cm.search_ee_data(
            self.search_keyword, regex=self.if_regex, source="ee"
        )

        if search_result:
            df = pd.DataFrame(search_result)

        return knext.Table.from_pandas(df)


############################################
# GEE Image node
############################################


@knext.node(
    name="GEE Image",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.output_binary(
    name="Image",
    description="The output binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Image info view",
    description="Showing a json view with the GEE Image info for the first row",
)
class GEEImage:
    """GEE Image.
    GEE Image node.
    """

    data_set_id = knext.StringParameter(
        "Data Set ID",
        "The data set ID for the GEE Image",
        default_value="LANDSAT/LE7_TOA_5YEAR/1999_2003",
    )

    def configure(self, configure_context):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        image = ee.Image(self.data_set_id)

        info = image.getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Image Info</h3><pre>{json_string}</pre>"""
        import pickle

        image_string = pickle.dumps(image)
        return image_string, knext.view_html(html)


############################################
# GEE Feature Collection node
############################################
@knext.node(
    name="GEE Feature Collection",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.output_binary(
    name="Feature Collection",
    description="The output binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Feature Collection info view",
    description="Showing a json view with the GEE Feature Collection info for the first row",
)
class GEEFeatureCollection:
    """GEE Feature Collection.
    GEE Feature Collection node.
    """

    data_set_id = knext.StringParameter(
        "Data Set ID",
        "The data set ID for the GEE Feature Collection",
        default_value="TIGER/2018/States",
    )

    def configure(self, configure_context):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        feature_collection = ee.FeatureCollection(self.data_set_id)

        info = feature_collection.first().getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Feature Collection Info</h3><pre>{json_string}</pre>"""
        import pickle

        feature_collection_string = pickle.dumps(feature_collection)
        return feature_collection_string, knext.view_html(html)


############################################
# Local shapefile to GEE Feature Collection node
############################################
@knext.node(
    name="Local Shapefile to GEE Feature Collection",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.output_binary(
    name="Feature Collection",
    description="The output binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Feature Collection info view",
    description="Showing a json view with the GEE Feature Collection info for the first row",
)
class LocalShapefileToGEEFeatureCollection:
    """Local Shapefile to GEE Feature Collection.
    Local Shapefile to GEE Feature Collection node.
    """

    local_shapefile = knext.StringParameter(
        "Local Shapefile",
        "The local shapefile path",
        default_value="",
    )

    def configure(self, configure_context):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee
        import geemap
        import os

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        local_shapefile = self.local_shapefile
        feature_collection = geemap.shp_to_ee(local_shapefile)

        info = feature_collection.first().getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Feature Collection Info</h3><pre>{json_string}</pre>"""

        import pickle

        feature_collection_string = pickle.dumps(feature_collection)
        return feature_collection_string, knext.view_html(html)


############################################
# export image node
############################################
@knext.node(
    name="Export Image",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
class ExportImage:
    """Export Image.
    Export Image node.
    """

    output_path = knext.StringParameter(
        "Output Path",
        "The path to export the image",
        default_value="export_image",
    )

    scale = knext.IntParameter(
        "Scale",
        "The scale to use for the classifier",
        default_value=30,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        import geemap
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        image = pickle.loads(input_binary)
        geemap.ee_export_image(
            image, filename=self.output_path, scale=self.scale, region=image.geometry()
        )
        return None


############################################
# export feature collection node
############################################
@knext.node(
    name="Export Feature Collection",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
class ExportFeatureCollection:
    """Export Feature Collection.
    Export Feature Collection node.
    """

    output_path = knext.StringParameter(
        "Output Path",
        "The path to export the feature collection",
        default_value="export_feature_collection.shp",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        import geemap
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        feature_collection = pickle.loads(input_binary)
        geemap.ee_to_shp(feature_collection, filename=self.output_path)
        return None


############################################
# Gee Feature Collection to GeoTable
############################################
@knext.node(
    name="GEE Feature Collection to GeoTable",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_table(
    name="GeoTable",
    description="The output table containing the GEE Feature Collection.",
)
class GEEFeatureCollectionToGeoTable:
    """GEE Feature Collection to GeoTable.
    GEE Feature Collection to GeoTable node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        import geemap
        import pandas as pd
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        feature_collection = pickle.loads(input_binary)
        gdf = geemap.ee_to_gdf(feature_collection)

        return knext.Table.from_pandas(gdf)


############################################
# GeoTable to GEE Feature Collection
############################################
@knext.node(
    name="GeoTable to GEE Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_table(
    name="GeoTable",
    description="The input table containing the GeoTable.",
)
@knext.output_binary(
    name="Feature Collection",
    description="The output binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Feature Collection info view",
    description="Showing a json view with the GEE Feature Collection info for the first row",
)
class GeoTableToGEEFeatureCollection:
    """GeoTable to GEE Feature Collection.
    GeoTable to GEE Feature Collection node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_table):
        import ee
        import geemap
        import pandas as pd
        import geopandas as gpd
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        df = input_table.to_pandas()
        gdf = gpd.GeoDataFrame(df)
        feature_collection = geemap.geopandas_to_ee(gdf)

        info = feature_collection.first().getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Feature Collection Info</h3><pre>{json_string}</pre>"""

        feature_collection_string = pickle.dumps(feature_collection)
        return feature_collection_string, knext.view_html(html)


############################################
# Local Tiff to GEE Image node
############################################
@knext.node(
    name="Local Tiff to GEE Image",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.output_binary(
    name="Image",
    description="The output binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Image info view",
    description="Showing a json view with the GEE Image info for the first row",
)
class LocalTiffToGEEImage:
    """Local Tiff to GEE Image.
    Local Tiff to GEE Image node.
    """

    local_tiff = knext.StringParameter(
        "Local Tiff",
        "The local tiff path",
        default_value="",
    )

    def configure(self, configure_context):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee
        import geemap
        import os

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        local_tiff = self.local_tiff
        image = geemap.tif_to_ee(local_tiff)

        info = image.getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Image Info</h3><pre>{json_string}</pre>"""
        import pickle

        image_string = pickle.dumps(image)
        return image_string, knext.view_html(html)


############################################
# Export Video node
############################################
# @knext.node(
#     name="Export Video",
#     node_type=knext.NodeType.SINK,
#     category=__category,
#     icon_path=__NODE_ICON_PATH + "GEE.png",
#     after="",
# )
# @knext.input_binary(
#     name="Image",
#     description="The input binary containing the GEE Image.",
#     id="geemap.gee.ImageCollection",
# )

# class ExportVideo:
#     """Export Video.
#     Export Video node.
#     """
#     output_path = knext.StringParameter(
#         "Output Path",
#         "The path to export the video",
#         default_value="export_video",
#     )

#     video_args = knext.StringParameter(
#         "Video Args",
#         "The video arguments",
#         default_value="{}",
#     )
#     def configure(self, configure_context, input_schema):
#         return None

#     def execute(self, exec_context: knext.ExecutionContext, input_binary):
#         import ee
#         import geemap
#         import pickle

#         ee.Authenticate()
#         ee.Initialize(project='gogletetst')
#         import json
#         video_args = json.loads(self.video_args)
#         images = pickle.loads(input_binary)
#         geemap.download_ee_video(images,
#                                  video_args,self.output_path,
#                                 )
#         return None


############################################
# Get Image Info node
############################################
@knext.node(
    name="Get Image Info",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Image",
    description="The output binary containing the GEE Image. The Same as the input Image.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Image info view",
    description="Showing a json view with the GEE Image info for the first row",
)
class GetImageInfo:
    """Get Image Info.
    Get Image Info node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        import geemap
        import pandas as pd
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        image = pickle.loads(input_binary)
        info = image.getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Image Info</h3><pre>{json_string}</pre>"""

        return input_binary, knext.view_html(html)


############################################
# Get Feature Collection Info node
############################################
@knext.node(
    name="Get Feature Collection Info",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GEE.png",
    after="",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Feature Collection",
    description="The output binary containing the GEE Feature Collection. The Same as the input Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_view(
    name="Feature Collection info view",
    description="Showing a json view with the GEE Feature Collection info for the first row",
)
class GetFeatureCollectionInfo:
    """Get Feature Collection Info.
    Get Feature Collection Info node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        import geemap
        import pandas as pd
        import pickle

        ee.Authenticate()
        ee.Initialize(project="gogletetst")
        feature_collection = pickle.loads(input_binary)
        info = feature_collection.first().getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>Feature Collection Info</h3><pre>{json_string}</pre>"""

        return input_binary, knext.view_html(html)
