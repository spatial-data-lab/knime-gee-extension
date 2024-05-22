import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="manipulator",
    name="Manipulator",
    description="Manipulator nodes for Google Earth Engine",
    icon="icons/manipulator.png",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/manipulator/"


############################################
# data set mean calculation node
############################################
@knext.node(
    name="Calculate Mean",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image Collection.",
    id="geemap.gee.ImageCollection",
)
@knext.output_binary(
    name="Mean",
    description="The output binary containing the mean of the GEE Image Collection.",
    id="geemap.gee.Image",
)
class CalculateMean:
    """Calculate Mean.
    Calculate Mean node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import pickle

        image_collection = pickle.loads(input_binary)
        mean = image_collection.mean()

        mean_string = pickle.dumps(mean)
        return mean_string


############################################
# Image Clip node
############################################
@knext.node(
    name="Clip Image",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Clipped Image",
    description="The output binary containing the clipped GEE Image.",
    id="geemap.gee.Image",
)
class ClipImage:
    """Clip Image.
    Clip Image node.
    """

    def configure(self, configure_context, input_schema, input_schema_1):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, input_binary, input_binary_1
    ):
        import pickle

        image = pickle.loads(input_binary)
        feature_collection = pickle.loads(input_binary_1)
        clipped_image = image.clip(feature_collection.geometry())

        clipped_image_string = pickle.dumps(clipped_image)
        return clipped_image_string


############################################
# ImageCollection filterBounds node
############################################
@knext.node(
    name="Filter Bounds",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="Image Collection",
    description="The input binary containing the GEE Image Collection.",
    id="geemap.gee.ImageCollection",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Filtered Image Collection",
    description="The output binary containing the filtered GEE Image Collection.",
    id="geemap.gee.ImageCollection",
)
class FilterBounds:
    """Filter Bounds.
    Filter Bounds node.
    """

    def configure(self, configure_context, input_schema, input_schema_1):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, input_binary, input_binary_1
    ):
        import pickle

        image_collection = pickle.loads(input_binary)
        feature_collection = pickle.loads(input_binary_1)
        filtered_image_collection = image_collection.filterBounds(
            feature_collection.geometry()
        )

        filtered_image_collection_string = pickle.dumps(filtered_image_collection)
        return filtered_image_collection_string


############################################
# Image sampleRegions node
############################################
@knext.node(
    name="Sample Regions",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Sampled Regions",
    description="The output binary containing the sampled regions from the GEE Image.",
    id="geemap.gee.Image",
)
class SampleRegions:
    """Sample Regions.
    Sample Regions node.
    """

    # properties_list = knut.get_properties_list()
    properties = knext.StringParameter(
        "Properties",
        "The properties to include in the sampled regions.",
        default_value="LC",
    )
    scale = knext.IntParameter(
        "Scale",
        "The scale to use for sampling.",
        default_value=30,
    )

    def configure(self, configure_context, input_schema, input_schema_1):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, input_binary, input_binary_1
    ):
        import ee
        import pickle

        image = pickle.loads(input_binary)
        feature_collection = pickle.loads(input_binary_1)
        sampled_regions = image.sampleRegions(
            collection=feature_collection,
            properties=[self.properties],
            scale=self.scale,
        )
        sampled_regions_string = pickle.dumps(sampled_regions)
        return sampled_regions_string


############################################
# FeatureCollection errorMatrix node
############################################
@knext.node(
    name="Error Matrix",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "manipulator.png",
    after="",
)
@knext.input_binary(
    name="Sampled Regions",
    description="The input binary containing the sampled regions.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="Error Matrix",
    description="The output binary containing the error matrix.",
    id="geemap.gee.errorMatrix",
)
@knext.output_view(
    name="Error Matrix View", description="Showing the accuracy of the classification."
)
class ErrorMatrix:
    """Error Matrix.
    Error Matrix node.
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import pickle

        sampled_regions = pickle.loads(input_binary)
        error_matrix = sampled_regions.errorMatrix("LC", "classification")
        overall_accuracy = error_matrix.accuracy().getInfo()
        consumer_accuracy = error_matrix.consumersAccuracy().getInfo()
        producer_accuracy = error_matrix.producersAccuracy().getInfo()
        kappa = error_matrix.kappa().getInfo()
        order = error_matrix.order().getInfo()
        output_text = f"Overall Accuracy: {overall_accuracy}\n\nConsumer Accuracy: {consumer_accuracy}\n\nProducer Accuracy: {producer_accuracy}\n\nKappa: {kappa}\n\nOrder: {order}"
        html_template = """
        <html>
        <head>
            <title>Error Matrix</title>
        </head>
        <body>
            <h1>Error Matrix</h1>
            <p>{output_text}</p>
        </body>
        </html>
        """
        output_html = html_template.format(output_text=output_text)
        error_matrix_string = pickle.dumps(error_matrix)
        return error_matrix_string, knext.view_html(output_html)
