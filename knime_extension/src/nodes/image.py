"""
GEE Image I/O Nodes for KNIME
This module contains nodes for reading, filtering, and processing Google Earth Engine Images.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
)

# Category for GEE Image I/O nodes
__category = knext.category(
    path="/community/gee",
    level_id="imageio",
    name="Image IO",
    description="Google Earth Engine Image Input/Output and Processing nodes",
    icon="icons/ImageIO.png",
    after="imagecollection",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/image/"


############################################
# GEE Image Reader
############################################


@knext.node(
    name="Image Reader",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageReader.png",
    id="imagereader",
    after="",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
class ImageReader:
    """Loads a single image from Google Earth Engine using the specified image ID.

    This node allows you to access individual satellite images, elevation data, or other geospatial datasets from GEE's
    extensive catalog for further analysis in KNIME workflows.

    **Common Image Examples:**

    - Elevation: 'USGS/SRTMGL1_003' (30m resolution)

    - ESA Elevation: 'ESA/WorldCover/v100' (10m resolution)

    - WorldPop Population: 'CIESIN/GPWv411/GPW_Population_Density' (30 arc-second)

    - Global Forest: 'UMD/hansen/global_forest_change_2021_v1_9' (30m resolution)

    - Global Settlement: 'WSF/WSF_v1' (10m resolution)
    """

    imagename = knext.StringParameter(
        "Image Name",
        "The name/ID of the GEE image to load (e.g., 'USGS/SRTMGL1_003')",
        default_value="USGS/SRTMGL1_003",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee

        image = ee.Image(self.imagename)

        return knut.export_gee_connection(image, gee_connection)


############################################
# Image Band Selector
############################################


@knext.node(
    name="Image Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandSelector.png",
    id="bandselector",
    after="imagereader",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with filtered image object.",
    port_type=google_earth_engine_port_type,
)
class ImageBandSelector:
    """Filters and selects specific bands from a Google Earth Engine image.

    This node allows you to filter and select specific bands from a Google Earth Engine image, allowing you to focus on relevant spectral information and reduce data size.
    This node is useful for preparing images for specific applications like vegetation analysis, water detection, or optimizing processing speed by selecting only necessary bands.

    **Common Band Combinations:**

    - **RGB**: 'B4,B3,B2' (Sentinel-2) or 'B4,B3,B2' (Landsat 8)

    - **False Color**: 'B8,B4,B3' (Sentinel-2) - good for vegetation

    - **SWIR**: 'B12,B8,B4' (Sentinel-2) - good for moisture detection

    - **NDVI Bands**: 'B8,B4' (Sentinel-2) - for vegetation index calculation


    **Note:** If no specified bands are found in the image, the original image is returned unchanged.
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of band names to select (e.g., 'B1,B2,B3'). Leave empty to keep all bands.",
        default_value="",
    )

    def configure(self, configure_context, input_binary_spec):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        image = image_connection.gee_object

        # Only process if bands parameter is not empty
        if self.bands.strip():
            # Get original image info for logging

            # Optimize: only get band names, not full image info
            original_bands = image.bandNames().getInfo()
            # LOGGER.warning(f"Original image bands: {original_bands}")

            band_list = [band.strip() for band in self.bands.split(",") if band.strip()]
        # LOGGER.warning(f"Selecting bands: {band_list}")

        # Filter to only include bands that exist in the image
        available_bands = [band for band in band_list if band in original_bands]
        if available_bands:
            image = image.select(available_bands)
        else:
            LOGGER.warning(
                f"No specified bands found in image. Available bands: {original_bands}"
            )

        return knut.export_gee_connection(image, image_connection)


############################################
# Image Get Info
############################################


@knext.node(
    name="Image Get Info",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageGetInfo.png",
    after="imagereader",
    id="imageinfo",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection (pass-through).",
    port_type=google_earth_engine_port_type,
)
@knext.output_view(
    name="Image Info View",
    description="HTML view showing detailed image information",
)
class ImageGetInfo:
    """Displays detailed information about a Google Earth Engine image.

    This node extracts and displays comprehensive metadata about a GEE image including
    band information, properties, geometry, and other technical details. This is useful
    for understanding image structure before processing, verifying band names and properties,
    and debugging image-related issues.

    **Information Displayed:**

    - **Band Information**: Names, types, and properties of all bands
    - **Image Properties**: Metadata, acquisition date, cloud cover, etc.
    - **Geometry**: Bounding box and projection information
    - **System Properties**: GEE internal properties and identifiers

    **Use Cases:**

    - Explore image structure and available bands
    - Verify image properties before analysis
    - Debug image processing issues
    - Understand image metadata and acquisition parameters
    """

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import json
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image from connection (GEE already initialized)
            image = image_connection.gee_object

            # Get image information efficiently
            info = image.getInfo()

            # Format as JSON for display
            json_string = json.dumps(info, indent=2)

            # Create HTML view
            html = f"""
            <div style="font-family: monospace; font-size: 12px;">
                <h3>Image Information</h3>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
                {json_string}
                </pre>
            </div>
            """

            LOGGER.warning("Successfully retrieved image information")
            return knut.export_gee_connection(image, image_connection), knext.view_html(
                html
            )

        except Exception as e:
            LOGGER.error(f"Failed to get image info: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Error</h3>
                <p>Failed to retrieve image information: {str(e)}</p>
            </div>
            """
            return knut.export_gee_connection(image, image_connection), knext.view_html(
                error_html
            )


############################################
# Image Exporter
############################################


@knext.node(
    name="Image Exporter",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ExportImage.png",
    id="imageexporter",
    after="imageinfo",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
class ImageExporter:
    """Exports a Google Earth Engine image to local storage.

    This node exports a GEE image to your local file system in various formats.
    It handles the export process efficiently and provides options for controlling
    the export resolution and file format.

    **Export Options:**

    - **Output Path**: Specify the local file path for the exported image
    - **Scale**: Control the export resolution (meters per pixel)
    - **Format**: Automatically determined by file extension

    **Supported Formats:**

    - **GeoTIFF**: .tif, .tiff (recommended for most use cases)
    - **JPEG**: .jpg, .jpeg (for visualization)
    - **PNG**: .png (for visualization)

    **Use Cases:**

    - Export processed images for further analysis in other software
    - Create high-resolution maps for presentations
    - Generate base maps for GIS applications
    - Export classification results or derived products

    **Performance Notes:**

    - Larger scale values (lower resolution) export faster
    - Very high resolution exports may take considerable time
    - Consider using appropriate scale for your analysis needs
    """

    output_path = knext.StringParameter(
        "Output Path",
        "Local file path for the exported image (include file extension, e.g., 'output.tif')",
        default_value="exported_image.tif",
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters per pixel for export (lower = higher resolution)",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import geemap
        import logging
        import os

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image from connection (GEE already initialized)
            image = image_connection.gee_object

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Export image using geemap
            geemap.ee_export_image(
                image,
                filename=self.output_path,
                scale=self.scale,
                region=image.geometry(),
            )

            LOGGER.warning(f"Successfully exported image to: {self.output_path}")
            LOGGER.warning(f"Export scale: {self.scale} meters per pixel")

        except Exception as e:
            LOGGER.error(f"Image export failed: {e}")
            raise


############################################
# GeoTiff To GEE Image
############################################


@knext.node(
    name="Local GeoTiff To GEE",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GeotiffToGEE.png",
    id="tifftogee",
    after="imageexporter",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=google_earth_engine_port_type,
)
@knext.output_view(
    name="Image Info View",
    description="HTML view showing imported image information",
)
class GeoTiffToGEEImage:
    """Converts a local GeoTIFF file to a Google Earth Engine image.

    This node uploads and converts a local GeoTIFF file to a GEE image object,
    making it available for processing within GEE workflows. This is useful for
    incorporating local data, custom analysis results, or external datasets into
    GEE-based analysis.

    **Supported Formats:**

    - **GeoTIFF**: .tif, .tiff files with geographic information
    - **Single Band**: Grayscale or single-band images
    - **Multi-Band**: RGB, multispectral, or hyperspectral images

    **Use Cases:**

    - Upload custom classification results for further analysis
    - Incorporate local survey data or field measurements
    - Import external satellite imagery not available in GEE
    - Upload processed results from other software for visualization

    **Important Notes:**

    - File must be in a supported geographic projection (preferably WGS84)
    - Large files may take time to upload and process
    - Ensure sufficient storage quota in your GEE account
    - File path must be accessible from the KNIME environment
    """

    local_tiff_path = knext.StringParameter(
        "Local GeoTIFF Path",
        "Full path to the local GeoTIFF file to upload",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import geemap
        import logging
        import json
        import os

        LOGGER = logging.getLogger(__name__)

        try:
            # Validate file path
            if not self.local_tiff_path or not os.path.exists(self.local_tiff_path):
                raise FileNotFoundError(
                    f"GeoTIFF file not found: {self.local_tiff_path}"
                )

            # Convert local GeoTIFF to GEE image
            image = geemap.tif_to_ee(self.local_tiff_path)

            # Get basic image information for display
            info = image.getInfo()
            json_string = json.dumps(info, indent=2)

            # Create HTML view
            html = f"""
            <div style="font-family: monospace; font-size: 12px;">
                <h3>Imported Image Information</h3>
                <p><strong>File:</strong> {os.path.basename(self.local_tiff_path)}</p>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
{json_string}
                </pre>
            </div>
            """

            LOGGER.warning(f"Successfully imported GeoTIFF: {self.local_tiff_path}")
            return knut.export_gee_connection(image, gee_connection), knext.view_html(
                html
            )

        except Exception as e:
            LOGGER.error(f"GeoTIFF import failed: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Import Error</h3>
                <p>Failed to import GeoTIFF: {str(e)}</p>
                <p><strong>File:</strong> {self.local_tiff_path}</p>
            </div>
            """
            # Return a dummy image connection to maintain workflow continuity
            dummy_image = ee.Image.constant(0)
            return knut.export_gee_connection(
                dummy_image, gee_connection
            ), knext.view_html(error_html)
