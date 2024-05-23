import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="authorization",
    name="Authorization",
    description="Authorization nodes for Google Earth Engine",
    icon="icons/authorization.png",
)

# # Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/authorization/"

# ############################################
# # GEE Authenticate node
# ############################################
@knext.node(
    name="GEE Authenticate",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "authorization.png",
    after="",
)
@knext.output_binary(
    name="GEE Authenticate",
    description="The output binary containing the GEE Authenticate.",
    id="geemap.gee.Authenticate",
)

class GEEAuthenticate:
    """GEE Authenticate.
    GEE Authenticate node.
    """
    # project_name = knext.StringParameter(
    #     "Project Name",
    #     "The project name to authenticate",
    #     default_value="gogletetst",
    # )
    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee
        ee.Authenticate()
        ee.Initialize()
        import pickle
        authenticate = "Authenticated"
        authenticate_string = pickle.dumps(authenticate)
        return authenticate_string
