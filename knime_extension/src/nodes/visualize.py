import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="visualize",
    name="Visualize",
    description="Visualize nodes for Google Earth Engine",
    icon="icons/visualize.png",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/visualize/"


############################################
# GEEMap View Node
############################################
@knext.node(
    name="GEE Map",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=__NODE_ICON_PATH + "visualize.png",
    category=__category,
    after="",
)
@knext.input_binary(
    name="GEE Map",
    description="A binary file containing the GEE map.",
    id="geemap.gee.Image",
)
# @knext.input_binary(
#     name="GEE Map 2",
#     description="A binary file containing the GEE map.",
#     id="geemap.gee.Image"


@knext.output_view(
    name="GEE Map View",
    description="Showing a map with the GEE map",
    static_resources="libs/leaflet/1.9.3",
)
class ViewNodeGEEMap:
    """Visualizes a GEE map on a map.

    This node will visualize the given GEE map on a map using the [geemap](https://geemap.org/) library.
    This view is highly interactive and allows you to change various aspects of the view within the visualization itself.
    For more information about the supported interactions see the [geemap user guides](https://geemap.org/).

    This node uses the [Leaflet](https://leafletjs.com/) library to display the map. The Leaflet library is a
    leading open-source JavaScript library for mobile-friendly interactive maps. It is designed with simplicity,
    performance, and usability in mind. For more information about the Leaflet library see the
    [Leaflet documentation](https://leafletjs.com/reference-1.7.1.html).

    This node requires the [geemap](https://geemap.org/) library to be installed. The geemap library is a Python package
    that provides interactive mapping and satellite imagery analysis using Google Earth Engine within Jupyter
    notebooks, JupyterLab, and the integrated development environment (IDE) of the Python language. For more information
    about the geemap library see the [geemap documentation](https://geemap.org/)."""

    viz_params = knext.StringParameter(
        "Visualization parameters",
        "A JSON string containing the visualization parameters for the GEE map. "
        "The visualization parameters are specific to the GEE map and can be found in the GEE documentation.",
        default_value="""{"min": 0.0, "max": 0.3, "bands": ["B4", "B3", "B2"]}""",
    )

    zoom = knext.IntParameter(
        "Zoom",
        "The zoom level of the map",
        default_value=12,
    )
    name = knext.StringParameter(
        "Name",
        "The name of the GEE map",
        default_value="GEE Map",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, input_binary):
        import ee
        ee.Authenticate()
        ee.Initialize()
        import pickle

        image = pickle.loads(input_binary)
        # image1 = pickle.loads(input_binary_1)

        import json

        # create a map
        vis = json.loads(self.viz_params)
        import geemap.foliumap as geemap

        Map = geemap.Map()
        Map.addLayer(image, vis, self.name)
        # if image1 is not None:
        #     Map.addLayer(image1,vis ,self.name)
        # center the map
        Map.centerObject(image, self.zoom)
        # replace css and JavaScript paths
        html = Map.get_root().render()
        # html = replace_external_js_css_paths(
        #     r'\1./libs/leaflet/1.9.3/\3"\4',
        #     html,
        # )

        return knext.view(html)
