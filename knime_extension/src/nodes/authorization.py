import knime_extension as knext
import util.knime_utils as knut
from google.oauth2.credentials import (
    Credentials,
)
from util.common import (
    GoogleEarthEngineObjectSpec,
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

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
    name="Google Earth Engine Connector",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "authorization.png",
    keywords=[
        "Google",
        "Google Earth Engine",
        "Satellite",
        "Google API Auth",
    ],
)
@knext.input_port(
    name="Google Connection",
    description="Google Earth Engine credentials.",
    port_type=knext.PortType.CREDENTIAL,
)
@knext.output_port(
    "Google Earth Engine Connection",
    "A connection to a Google Earth Engine account.",
    google_earth_engine_port_type,
)
class GEEAuthenticate:
    """GEE Authenticate.
    GEE Authenticate node.
    """

    project_id = knext.StringParameter(
        label="Project Id",
        description=("The client project ID or number to use when making API calls."),
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        credential: knext.PortObject,
    ):
        # Combine credentials with customer ID
        # Use the access token provided in the input port.
        # Token refresh is handled by the provided refresh handler that requests the token from the input port.
        credentials = Credentials(
            token=str(credential.spec.auth_parameters),
            expiry=credential.spec.expires_after,
            refresh_handler=get_refresh_handler(credential.spec),
        )

        import ee

        # ee.Authenticate(authorization_code=credentials)
        ee.Initialize(credentials=credentials, project=self.project_id)
        # Create a client for Google Earth Engine
        client = ee.data

        port_object = GoogleEarthEngineConnectionObject(
            GoogleEarthEngineObjectSpec(account_id="test"),
            client=client,
        )
        return port_object


def get_refresh_handler(
    spec: knext.CredentialPortObjectSpec,
) -> callable:
    """Returns a function that returns the access token and the expiration time of the access token."""
    return lambda request, scopes: (
        spec.auth_parameters,
        spec.expires_after,
    )
