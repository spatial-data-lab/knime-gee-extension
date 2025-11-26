import ee
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
    icon_path=__NODE_ICON_PATH + "Authenticate.png",
    keywords=[
        "Google",
        "Earth Engine",
        "Google Earth Engine",
        "Google Earth Engine",
        "Satellite",
        "Geospatial raster data",
        "Google API Authentication",
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
    """Establishes a connection to Google Earth Engine (GEE) with Google Authenticator.

    This node establishes a connection to Google Earth Engine (GEE) for accessing satellite imagery
    and geospatial data. To get started with cloud based remote sensing with the Google Earth Engine check out
    the Fundamentals and Application book at [https://www.eefabook.org/](https://www.eefabook.org/).

    The node supports two authentication methods with Google Authenticator:

    1. **Interactive Authentication**: Uses your personal Google account credentials
    2. **Service Account Authentication**: Uses a Google Cloud service account JSON key

    **Setup Instructions:**

    **For Interactive Authentication:**

    - In the Google Authenticator node, select "Interactive" authentication method
    - Set scope to "Custom" and enter: *https://www.googleapis.com/auth/earthengine*
    - Sign in with your Google account that has Earth Engine access

    **For Service Account Authentication:**

    - In the Google Authenticator node, select "Google Service Account JSON Key" method
    - Upload your service account JSON key file
    - Set scope to "Custom" and enter: *https://www.googleapis.com/auth/earthengine*

    **Important**: You must add the "Service Usage Consumer" role to your service account:

      - Go to Google Cloud Console > IAM & Admin > IAM
      - Find your service account and click the Edit icon
      - Click "Add another role"
      - Search for "Service Usage Consumer" and add it
      - Click "Save" and wait a few minutes for activation

    **Project ID:**

    - Must be associated with either your interactive account or service account
    - **Easy way to find it**: Visit [Earth Engine Code Editor](https://code.earthengine.google.com/) and click the
        avatar in the top-right corner to view Project info or follow the instructions from the
        [Google documentation.](https://support.google.com/googleapi/answer/7014113)
    - The Project ID is required for all GEE operations

    **Note:** Ensure your Google account or service account has been approved for Earth Engine access
    at [Earth Engine signup.](https://signup.earthengine.google.com/)
    """

    project_id = knext.StringParameter(
        label="Project ID",
        description=(
            """The Google Cloud Project ID associated with your Earth Engine account. 
            To find your project id visit [Earth Engine Code Editor](https://code.earthengine.google.com/) 
            and click the avatar in the top-right corner to view Project info or follow the instructions from the
            [Google documentation.](https://support.google.com/googleapi/answer/7014113)
            """
        ),
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        credential: knext.PortObject,
    ):

        from google.oauth2.credentials import (
            Credentials,
        )

        # Combine credentials with customer ID
        # Use the access token provided in the input port.
        # Token refresh is handled by the provided refresh handler that requests the token from the input port.
        credential_spec = credential.spec
        credentials = Credentials(
            token=str(credential_spec.auth_parameters),
            expiry=credential_spec.expires_after,
            refresh_handler=get_refresh_handler(credential_spec),
        )

        import ee

        exec_context.set_progress(
            0.5, f"Initializing GEE with project ID: {self.project_id}"
        )

        knut.check_canceled(exec_context)

        # Initialize the Earth Engine API with the provided credentials and project ID
        ee.Initialize(credentials=credentials, project=self.project_id)

        port_object = GoogleEarthEngineConnectionObject(
            GoogleEarthEngineObjectSpec(project_id=self.project_id),
            credentials=credentials,
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
