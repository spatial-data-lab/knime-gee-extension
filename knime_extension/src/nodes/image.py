"""
GEE Image I/O Nodes for KNIME
This module contains nodes for reading, filtering, and processing Google Earth Engine Images.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
    gee_feature_collection_port_type,
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
    name="EE Image Reader",
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
    port_type=gee_image_port_type,
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

        return knut.export_gee_image_connection(image, gee_connection)


############################################
# Image Band Selector
############################################


@knext.node(
    name="EE Image Band Selector",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandSelector.png",
    id="bandselector",
    after="imagereader",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with filtered image object.",
    port_type=gee_image_port_type,
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
        image = image_connection.image

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

        return knut.export_gee_image_connection(image, image_connection)


############################################
# Image Get Info
############################################


@knext.node(
    name="EE Image Get Info",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageGetInfo.png",
    after="imagereader",
    id="imageinfo",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
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
            image = image_connection.image

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
            return knext.view_html(html)

        except Exception as e:
            LOGGER.error(f"Failed to get image info: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Error</h3>
                <p>Failed to retrieve image information: {str(e)}</p>
            </div>
            """
            return knext.view_html(error_html)


############################################
# Image Clip
############################################


@knext.node(
    name="EE Image Clip",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageClip.png",
    id="imageclip",
    after="imageinfo",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="Feature Collection defining the region to clip the image to.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with clipped image object.",
    port_type=gee_image_port_type,
)
class ImageClip:
    """Clips a Google Earth Engine image to a specific geometry.

    This node clips a GEE image to the boundaries of a provided geometry,
    which can be a feature collection, feature, or geometry object. This is useful
    for focusing analysis on specific regions of interest and reducing processing time
    and data size.

    **Use Cases:**

    - Extract image data for a specific study area
    - Prepare images for export by limiting to area of interest
    - Reduce processing time by clipping to smaller regions
    - Create regional mosaics or composites

    **Input Geometry Types:**

    - **Feature Collection**: Uses the geometry of the entire collection
    - **Feature**: Uses the geometry of the feature
    - **Geometry**: Uses the geometry directly

    **Performance Notes:**

    - Clipping reduces the area to process, improving performance
    - Especially useful before exporting images
    - Can significantly reduce export file size
    """

    def configure(self, configure_context, input_schema_1, input_schema_2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        geometry_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image from connection
        image = image_connection.image

        # Get geometry from Feature Collection connection
        feature_collection = geometry_connection.feature_collection

        # Extract geometry from Feature Collection
        geometry = feature_collection.geometry()

        # Clip the image
        clipped_image = image.clip(geometry)

        LOGGER.warning("Successfully clipped image to geometry")
        return knut.export_gee_image_connection(clipped_image, image_connection)


############################################
# Image Exporter
############################################


@knext.node(
    name="EE Image Exporter",
    node_type=knext.NodeType.SINK,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ExportImage.png",
    id="imageexporter",
    after="imageclip",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
class ImageExporter:
    """Exports a Google Earth Engine image to Google Drive or Cloud Storage.

    This node submits an Earth Engine export task that writes the image either
    to a Google Cloud Storage bucket or to Google Drive. You can optionally wait
    for the export task to finish and monitor progress through the node logs.

    **Destinations:**

    - **Google Cloud Bucket**: Provide a bucket/object path such as
      ``my-bucket/folder/my_image.tif``.
    - **Google Drive**: Provide a folder and file name such as ``EEexport/my_image.tiff``.

    **Authentication Notes:**

    - When using a **service account**, choose Google Cloud Bucket. Service accounts
      usually do not have Google Drive scopes or permissions.
    - When using **interactive authentication**, add the appropriate scopes in the
      Google Authenticator node (Drive or Cloud Storage) before selecting the destination.

    **Options:**

    - **Scale**: Controls the export resolution in meters per pixel.
    - **Wait for Download Completion**: When enabled, the node polls the export
      task until it succeeds or fails.

    All exports use the GeoTIFF format with one file containing all bands.
    """

    class DestinationModeOptions(knext.EnumParameterOptions):
        CLOUD = (
            "Google Cloud Bucket",
            "Export image to a Google Cloud Storage bucket.",
        )
        DRIVE = ("Google Drive", "Export image to a Google Drive folder.")

        @classmethod
        def get_default(cls):
            return cls.CLOUD

    destination = knext.EnumParameter(
        label="Destination Mode",
        description="Choose between exporting to Google Drive or Google Cloud Storage.",
        default_value=DestinationModeOptions.get_default().name,
        enum=DestinationModeOptions,
    )

    drive_path = knext.StringParameter(
        "Drive Path",
        "Drive folder and file name, e.g., 'EEexport/my_image.tiff'.",
        default_value="EEexport/export.tiff",
    ).rule(
        knext.OneOf(destination, [DestinationModeOptions.DRIVE.name]),
        knext.Effect.SHOW,
    )

    cloud_object_path = knext.StringParameter(
        "Cloud Storage Object",
        "Bucket and object path, e.g., 'my-bucket/folder/my_image.tif'.",
        default_value="bucket/export.tif",
    ).rule(
        knext.OneOf(destination, [DestinationModeOptions.CLOUD.name]),
        knext.Effect.SHOW,
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters per pixel for export (lower = higher resolution)",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    wait_for_completion = knext.BoolParameter(
        "Wait for Download Completion",
        "If enabled, the node will wait until the export finishes before completing.",
        default_value=True,
    )

    max_pixels = knext.IntParameter(
        "Max Pixels",
        "Maximum number of pixels Earth Engine is allowed to read during export (integer ≤ 2,147,483,647).",
        default_value=1000000000,
        min_value=1,
        is_advanced=True,
    )

    max_wait_seconds = knext.IntParameter(
        "Max Wait Seconds",
        "Maximum number of seconds to wait for completion when waiting is enabled.",
        default_value=600,
        min_value=1,
        is_advanced=True,
    ).rule(
        knext.OneOf(wait_for_completion, [True]),
        knext.Effect.SHOW,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import logging
        import os
        import time
        from datetime import datetime

        LOGGER = logging.getLogger(__name__)

        image = image_connection.image
        destination = self.destination or self.DestinationModeOptions.get_default().name

        try:
            destination_option = self.DestinationModeOptions[destination]
        except KeyError as exc:
            valid_options = ", ".join(opt.name for opt in self.DestinationModeOptions)
            raise ValueError(
                f"Unsupported destination '{self.destination}'. Expected one of [{valid_options}]."
            ) from exc

        LOGGER.info(
            "Starting image export: destination=%s, scale=%s",
            destination_option.value[0],
            self.scale,
        )

        region = image.geometry()

        try:
            projection = image.projection().getInfo()
            LOGGER.warning("Projection: %s", projection)
        except Exception as projection_error:
            LOGGER.warning("Failed to retrieve image projection: %s", projection_error)

        def wait_for_task(task, label, success_message):
            max_wait_seconds = self.max_wait_seconds
            check_interval = 5
            start_time = time.time()

            while True:
                status = task.status()
                state = status.get("state")

                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    if state == "COMPLETED":
                        LOGGER.warning(success_message)
                        return

                    LOGGER.error(
                        "%s export task %s ended with state %s: %s",
                        label,
                        task.id,
                        state,
                        status.get("error_message"),
                    )
                    raise RuntimeError(
                        f"{label} export failed with state {state}: "
                        f"{status.get('error_message')}"
                    )

                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_seconds:
                    LOGGER.error(
                        "Timed out waiting for %s export task %s (last state %s)",
                        label,
                        task.id,
                        state,
                    )
                    raise TimeoutError(
                        f"{label} export task {task.id} timed out before completion."
                    )

                LOGGER.warning(
                    "Waiting for %s export task %s... state=%s (elapsed %.1fs)",
                    label,
                    task.id,
                    state,
                    elapsed_time,
                )
                time.sleep(check_interval)

        if destination_option is self.DestinationModeOptions.CLOUD:
            cloud_path = (self.cloud_object_path or "").strip()
            if not cloud_path:
                raise ValueError(
                    "Cloud Storage Object must be provided when Destination is 'cloud'."
                )

            cloud_path = cloud_path.lstrip("/")
            full_gs_path = f"gs://{cloud_path}"
            full_gs_path = knut.ensure_file_extension(full_gs_path, ".tif")

            bucket_and_object = full_gs_path[5:]
            if "/" not in bucket_and_object:
                raise ValueError(
                    "Cloud Storage path must include both bucket and object "
                    "(e.g., 'my-bucket/path/to/file.tif')."
                )

            bucket, object_path = bucket_and_object.split("/", 1)
            object_path = object_path.strip("/")
            if not object_path:
                raise ValueError(
                    "Cloud Storage path must include an object name after the bucket."
                )

            object_prefix, _ = os.path.splitext(object_path)

            description = f"KNIME Image Cloud Export {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            try:
                export_task = ee.batch.Export.image.toCloudStorage(
                    image=image,
                    description=description,
                    bucket=bucket,
                    fileNamePrefix=object_prefix,
                    region=region,
                    scale=self.scale,
                    maxPixels=self.max_pixels,
                    fileFormat="GeoTIFF",
                )
            except Exception as task_error:
                LOGGER.error(
                    "Failed to create Cloud Storage export task: %s", task_error
                )
                raise

            export_task.start()
            LOGGER.warning(
                "Started Cloud Storage export task %s -> gs://%s/%s*.tif",
                export_task.id,
                bucket,
                object_prefix,
            )

            if not self.wait_for_completion:
                LOGGER.warning(
                    "Cloud export submitted; not waiting for completion because "
                    "'Wait for Download Completion' is disabled."
                )
                return

            wait_for_task(
                export_task,
                "Cloud Storage",
                f"GCS export task {export_task.id} completed. Output prefix gs://{bucket}/{object_prefix}",
            )
            return

        drive_path = (self.drive_path or "").strip().lstrip("/")
        if not drive_path:
            raise ValueError("Drive Path must be provided when Destination is 'drive'.")

        parts = [p for p in drive_path.split("/") if p]
        if not parts:
            raise ValueError("Drive Path must include a file name.")

        filename = knut.ensure_file_extension(parts[-1], ".tiff")
        file_prefix, _ = os.path.splitext(filename)
        folder = "/".join(parts[:-1]) if len(parts) > 1 else None

        description = (
            f"KNIME Image Drive Export {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        export_kwargs = dict(
            image=image,
            description=description,
            fileNamePrefix=file_prefix,
            region=region,
            scale=self.scale,
            maxPixels=self.max_pixels,
            fileFormat="GeoTIFF",
        )
        if folder:
            export_kwargs["folder"] = folder

        try:
            export_task = ee.batch.Export.image.toDrive(**export_kwargs)
        except Exception as task_error:
            LOGGER.error("Failed to create Drive export task: %s", task_error)
            raise

        export_task.start()
        folder_label = folder or "root"
        LOGGER.warning(
            "Started Drive export task %s -> folder '%s', prefix '%s'",
            export_task.id,
            folder_label,
            file_prefix,
        )

        if not self.wait_for_completion:
            LOGGER.warning(
                "Drive export submitted; not waiting for completion because "
                "'Wait for Download Completion' is disabled."
            )
            return

        wait_for_task(
            export_task,
            "Drive",
            f"Drive export task {export_task.id} completed. Folder '{folder_label}', prefix '{file_prefix}'",
        )
        return


############################################
# GeoTiff To GEE Image
############################################


@knext.node(
    name="Cloud GeoTiff To EE Image",
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
    port_type=gee_image_port_type,
)
@knext.output_view(
    name="Image Info View",
    description="HTML view showing imported image information",
)
class GeoTiffToGEEImage:
    """Loads a GeoTIFF stored in Google Cloud Storage into Google Earth Engine.

    This node reads a GeoTIFF that already resides in a Google Cloud Storage bucket,
    creating an ``ee.Image`` that can be used in subsequent GEE nodes. The bucket
    must be accessible to the authenticated Earth Engine account (same project or
    granted read permissions).

    **Supported Formats:**

    - **GeoTIFF**: .tif, .tiff files with geographic information
    - **Single Band**: Grayscale or single-band images
    - **Multi-Band**: RGB, multispectral, or hyperspectral images

    **Use Cases:**

    - Bring Cloud Storage exports back into GEE workflows
    - Share processed rasters across projects via Cloud Storage
    - Avoid local file transfers when working entirely in the cloud

    **Important Notes:**

    - Provide the path as ``bucket/folder/file.tif`` or ``gs://bucket/folder/file.tif``
    - The authenticated account must have read access to the bucket/object
    - The GeoTIFF should be Cloud Optimized (recommended for faster access)
    """

    cloud_tiff_path = knext.StringParameter(
        "Cloud Storage GeoTIFF Path",
        "Path to the GeoTIFF in Google Cloud Storage (e.g., 'my-bucket/folder/file.tif').",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import logging
        import json
        import os

        LOGGER = logging.getLogger(__name__)

        cloud_path = (self.cloud_tiff_path or "").strip()
        if not cloud_path:
            raise ValueError(
                "Please provide the Cloud Storage path to the GeoTIFF (e.g., 'bucket/folder/file.tif')."
            )

        if not cloud_path.startswith("gs://"):
            cloud_path = f"gs://{cloud_path.lstrip('/')}"
        if not cloud_path.lower().endswith(".tif"):
            cloud_path = knut.ensure_file_extension(cloud_path, ".tif")

        LOGGER.info(f"Loading GeoTIFF from Cloud Storage: {cloud_path}")

        try:
            image = ee.Image.loadGeoTIFF(cloud_path)
        except Exception as exc:
            LOGGER.error(f"Failed to load GeoTIFF from Cloud Storage: {exc}")
            raise

        # Get basic image information for display
        info = image.getInfo()
        json_string = json.dumps(info, indent=2)

        # Create HTML view
        html = f"""
        <div style="font-family: monospace; font-size: 12px;">
            <h3>Imported Image Information</h3>
            <p><strong>Path:</strong> {cloud_path}</p>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
            {json_string}
            </pre>
        </div>
        """

        # LOGGER.warning(f"Successfully imported GeoTIFF: {self.local_tiff_path}")
        return knut.export_gee_image_connection(image, gee_connection), knext.view_html(
            html
        )
