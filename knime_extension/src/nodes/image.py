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
    name="GEE Image Reader",
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

    Visit [GEE Datasets Catalog](https://developers.google.com/earth-engine/datasets/catalog) to explore available
    datasets and their image IDs.

    **Common Image Examples:**

    - [Elevation](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003): 'USGS/SRTMGL1_003' (30m resolution)
    - [ESA Elevation](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100): 'ESA/WorldCover/v100' (10m resolution)
    - [WorldPop Population](https://developers.google.com/earth-engine/datasets/catalog/CIESIN_GPWv411_GPW_Population_Density): 'CIESIN/GPWv411/GPW_Population_Density' (30 arc-second)
    - [Global Forest](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2021_v1_9): 'UMD/hansen/global_forest_change_2021_v1_9' (30m resolution)
    - [Global Settlement](https://developers.google.com/earth-engine/datasets/catalog/DLR_WSF_WSF2015_v1): 'DLR/WSF/WSF2015/v1' (10m resolution)
    """

    imagename = knext.StringParameter(
        "Image name",
        """The name/ID of the GEE image to load (e.g., 'USGS/SRTMGL1_003')
        You can use the GEE Dataset Search node to find available images or 
        visit [GEE Datasets Catalog](https://developers.google.com/earth-engine/datasets/catalog).
        """,
        default_value="USGS/SRTMGL1_003",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, gee_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Validate image name
        imagename = (self.imagename or "").strip()
        if not imagename:
            raise ValueError(
                "Image name cannot be empty. Please provide a valid GEE image ID "
            )
        try:
            image = ee.Image(imagename)
        except Exception as e:
            LOGGER.error(f"Failed to load image '{imagename}': {e}")
            raise ValueError(
                f"Failed to load image '{imagename}'. Please verify the image ID is correct. "
                f"Error: {str(e)}"
            ) from e

        LOGGER.warning(f"Successfully loaded image: {imagename}")
        return knut.export_gee_image_connection(image, gee_connection)


############################################
# Image Band Selector
############################################


@knext.node(
    name="GEE Image Band Selector",
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

    This node allows you to filter and select specific bands from a Google Earth Engine image, allowing you to focus
    on relevant spectral information and reduce data size. This node is useful for preparing images for specific
    applications like vegetation analysis, water detection, or optimizing processing speed by selecting only necessary
    bands.

    Available bands vary by image source and can be explored using the **GEE Image Info Extractor** node.

    **Common Band Combinations:**

    - **RGB**: 'B4,B3,B2' (Sentinel-2) or 'B4,B3,B2' (Landsat 8)
    - **False Color**: 'B8,B4,B3' (Sentinel-2) - good for vegetation
    - **SWIR**: 'B12,B8,B4' (Sentinel-2) - good for moisture detection
    - **NDVI Bands**: 'B8,B4' (Sentinel-2) - for vegetation index calculation


    **Note:** If no specified bands are found in the image, the original image is returned unchanged.
    """

    bands = knext.StringParameter(
        "Bands",
        """Comma-separated list of band names to select (e.g., 'B1,B2,B3'). Leave empty to keep all bands.
        Available bands can be explored using the **GEE Image Info Extractor** node.""",
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
# Image Band Renamer
############################################


@knext.node(
    name="GEE Image Band Renamer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "bandrenamer.png",
    id="bandrenamer",
    after="bandselector",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with bands to rename.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with renamed bands.",
    port_type=gee_image_port_type,
)
class ImageBandRenamer:
    """Renames bands in a Google Earth Engine image.

    This node allows you to rename bands in an image, which is essential for
    organizing multi-temporal data and preparing bands for merging.

    **Use Cases:**

    - Rename bands to reflect time periods (e.g., 'stable_lights' → '2003')
    - Standardize band names across different images
    - Prepare bands for multi-temporal analysis

    **Examples:**

    - Rename single band: 'stable_lights' → '2003'
    - Rename multiple bands: 'B1,B2,B3' → 'Red,Green,Blue'

    **Note:** Number of new names must match number of bands to rename.
    If bands_to_rename is empty, all bands will be renamed.
    """

    bands_to_rename = knext.StringParameter(
        "Bands to rename",
        """Comma-separated list of band names to rename (e.g., 'stable_lights' or 'B1,B2,B3').
        Leave empty to rename all bands. If empty, new names must match total number of bands.""",
        default_value="",
    )

    new_names = knext.StringParameter(
        "New band names",
        """Comma-separated list of new names (e.g., '2003' or 'Red,Green,Blue').
        Number of names must match number of bands to rename.""",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        image = image_connection.image

        try:
            # Get all band names
            all_bands = image.bandNames().getInfo()

            # Determine which bands to rename
            if self.bands_to_rename.strip():
                bands_to_rename_list = [
                    b.strip() for b in self.bands_to_rename.split(",") if b.strip()
                ]
                # Validate bands exist
                missing_bands = [b for b in bands_to_rename_list if b not in all_bands]
                if missing_bands:
                    raise ValueError(f"Bands not found in image: {missing_bands}")
                bands_to_process = bands_to_rename_list
            else:
                # Rename all bands
                bands_to_process = all_bands

            # Get new names
            if not self.new_names.strip():
                raise ValueError("New band names must be provided")

            new_names_list = [n.strip() for n in self.new_names.split(",") if n.strip()]

            # Validate counts match
            if len(new_names_list) != len(bands_to_process):
                raise ValueError(
                    f"Number of new names ({len(new_names_list)}) must match "
                    f"number of bands to rename ({len(bands_to_process)})"
                )

            # Rename bands - start with first band
            renamed_image = image.select([bands_to_process[0]]).rename(
                new_names_list[0]
            )

            # Add remaining renamed bands
            for old_name, new_name in zip(bands_to_process[1:], new_names_list[1:]):
                renamed_band = image.select([old_name]).rename(new_name)
                renamed_image = renamed_image.addBands(renamed_band)

            # Add any bands that weren't renamed
            other_bands = [b for b in all_bands if b not in bands_to_process]
            if other_bands:
                other_image = image.select(other_bands)
                renamed_image = other_image.addBands(renamed_image)

            # Get final band names for logging
            final_bands = renamed_image.bandNames().getInfo()
            LOGGER.warning(
                f"Successfully renamed bands. Original: {bands_to_process}, "
                f"New: {new_names_list}, Final bands: {final_bands}"
            )

            return knut.export_gee_image_connection(renamed_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Failed to rename bands: {e}")
            raise


############################################
# Image Band Merger
############################################


@knext.node(
    name="GEE Image Band Merger",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "bandmerger.png",
    id="bandmerger",
    after="bandrenamer",
)
@knext.input_port(
    name="Base Image",
    description="Base image to add bands to. This will be the first image in the merged result.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Additional Image",
    description="Image containing bands to add to the base image.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with merged bands from both input images.",
    port_type=gee_image_port_type,
)
class ImageBandMerger:
    """Merges bands from two images into a single image.

    This node combines bands from two images using Google Earth Engine's
    addBands() method. You can chain multiple Band Merger nodes to merge
    more than two images.

    **Use Cases:**

    - Combine multi-temporal data (e.g., nighttime lights from different years)
    - Stack bands from different sources or time periods
    - Create time series stacks for change detection
    - Merge computed indices with original imagery

    **Examples:**

    - Merge 1993, 2003, 2013 nighttime lights:
      First merge: 2013 (base) + 2003 (additional) → merged_2013_2003
      Second merge: merged_2013_2003 (base) + 1993 (additional) → final merged image
    - Combine NDVI calculations with original spectral bands

    **Workflow Pattern:**

    1. Use **Band Selector** to select desired bands from each image
    2. Use **Band Renamer** to rename bands (e.g., 'stable_lights' → '2003')
    3. Use **Band Merger** to combine images (chain multiple nodes for 3+ images)
    """

    overwrite = knext.BoolParameter(
        "Overwrite existing bands",
        "If True, overwrite bands with the same name. If False, keep both bands (second one gets suffix).",
        default_value=False,
    )

    def configure(self, configure_context, base_image_schema, additional_image_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        base_image_connection,
        additional_image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        base_image = base_image_connection.image

        try:
            # Get band names from both images for logging
            base_bands = base_image.bandNames().getInfo()
            additional_image = additional_image_connection.image
            additional_bands = additional_image.bandNames().getInfo()

            # Add bands from additional image to base image
            merged_image = base_image.addBands(
                additional_image, overwrite=self.overwrite
            )

            # Get final band names
            final_bands = merged_image.bandNames().getInfo()
            LOGGER.warning(
                f"Successfully merged images. Base bands: {base_bands}, "
                f"Additional bands: {additional_bands}, Final bands: {final_bands}"
            )

            return knut.export_gee_image_connection(merged_image, base_image_connection)

        except Exception as e:
            LOGGER.error(f"Failed to merge image bands: {e}")
            raise


############################################
# Image Band Calculator
############################################


class BandOperationOptions(knext.EnumParameterOptions):
    """Options for band calculation operations."""

    ADDITION = ("Addition", "B1 + B2: Add two bands together")
    SUBTRACTION = ("Subtraction", "B1 - B2 or B2 - B1: Subtract one band from another")
    MULTIPLICATION = ("Multiplication", "B1 * B2: Multiply two bands")
    DIVISION = ("Division", "B1 / B2 or B2 / B1: Divide one band by another")
    NORMALIZED_DIFFERENCE = (
        "Normalized Difference",
        "(B1 - B2) / (B1 + B2): Commonly used for indices like NDVI, NDWI",
    )
    POWER = ("Power", "B1 ^ B2 or B2 ^ B1: Raise one band to the power of another")
    MAXIMUM = ("Maximum", "max(B1, B2): Maximum value of the two bands")
    MINIMUM = ("Minimum", "min(B1, B2): Minimum value of the two bands")
    MEAN = ("Mean", "(B1 + B2) / 2: Average of the two bands")

    @classmethod
    def get_default(cls):
        return cls.NORMALIZED_DIFFERENCE


@knext.node(
    name="GEE Image Band Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "bandcalculator.png",
    id="bandcalculator",
    after="bandmerger",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with bands to calculate from.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with calculated band added.",
    port_type=gee_image_port_type,
)
class ImageBandCalculator:
    """Performs mathematical operations between two bands and adds the result as a new band.

    This node performs basic mathematical operations between two bands of the same image
    and adds the calculated result as a new band. This is useful for creating custom
    indices, ratios, and band combinations.

    **Supported Operations:**

    - **Addition (+):** B1 + B2
    - **Subtraction (-):** B1 - B2 or B2 - B1
    - **Multiplication (*):** B1 * B2
    - **Division (/):** B1 / B2 or B2 / B1
    - **Normalized Difference:** (B1 - B2) / (B1 + B2) - commonly used for indices like NDVI
    - **Power (^):** B1 ^ B2 or B2 ^ B1
    - **Maximum:** max(B1, B2)
    - **Minimum:** min(B1, B2)
    - **Mean:** (B1 + B2) / 2

    **Common Use Cases:**

    - Calculate custom vegetation indices
    - Create band ratios (e.g., NIR/Red ratio)
    - Compute band differences for change detection
    - Generate composite indices for classification

    **Examples:**

    - NDVI-like calculation: bands="B8,B4", operation="Normalized Difference", output="ndvi"
    - NIR/Red ratio: bands="B8,B4", operation="Division" (B8/B4), output="nir_red_ratio"
    - Band difference: bands="B8,B4", operation="Subtraction" (B8-B4), output="nir_red_diff"
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of exactly 2 band names (e.g., 'B8,B4'). The order matters for subtraction and division.",
        default_value="B8,B4",
    )

    operation = knext.EnumParameter(
        label="Operation",
        description="Mathematical operation to perform between the two bands",
        default_value=BandOperationOptions.get_default().name,
        enum=BandOperationOptions,
    )

    reverse_order = knext.BoolParameter(
        "Reverse band order",
        "For subtraction, division, and power: if False, uses B1 - B2, B1 / B2, or B1 ^ B2; if True, uses B2 - B1, B2 / B1, or B2 ^ B1. Ignored for other operations.",
        default_value=False,
    ).rule(
        knext.OneOf(
            operation,
            [
                BandOperationOptions.SUBTRACTION.name,
                BandOperationOptions.DIVISION.name,
                BandOperationOptions.POWER.name,
            ],
        ),
        knext.Effect.SHOW,
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name for the calculated band that will be added to the image",
        default_value="calculated_band",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image from connection
        image = image_connection.image

        # Parse band names
        band_list = [b.strip() for b in self.bands.split(",") if b.strip()]

        # Validate exactly 2 bands
        if len(band_list) != 2:
            raise ValueError(
                f"Exactly 2 bands are required. Found {len(band_list)}: {band_list}"
            )

        band1_name, band2_name = band_list[0], band_list[1]

        # Get available bands
        available_bands = image.bandNames().getInfo()

        # Validate bands exist
        if band1_name not in available_bands:
            raise ValueError(
                f"Band '{band1_name}' not found in image. Available bands: {available_bands}"
            )
        if band2_name not in available_bands:
            raise ValueError(
                f"Band '{band2_name}' not found in image. Available bands: {available_bands}"
            )

        # Select the two bands
        band1 = image.select(band1_name)
        band2 = image.select(band2_name)

        # Perform calculation based on operation
        try:
            operation_name = self.operation

            if operation_name == BandOperationOptions.ADDITION.name:
                result = band1.add(band2)
            elif operation_name == BandOperationOptions.SUBTRACTION.name:
                if self.reverse_order:
                    result = band2.subtract(band1)
                else:
                    result = band1.subtract(band2)
            elif operation_name == BandOperationOptions.MULTIPLICATION.name:
                result = band1.multiply(band2)
            elif operation_name == BandOperationOptions.DIVISION.name:
                if self.reverse_order:
                    result = band2.divide(band1)
                else:
                    result = band1.divide(band2)
            elif operation_name == BandOperationOptions.NORMALIZED_DIFFERENCE.name:
                # Always use (B1 - B2) / (B1 + B2)
                result = image.normalizedDifference([band1_name, band2_name])
            elif operation_name == BandOperationOptions.POWER.name:
                if self.reverse_order:
                    result = band2.pow(band1)
                else:
                    result = band1.pow(band2)
            elif operation_name == BandOperationOptions.MAXIMUM.name:
                result = band1.max(band2)
            elif operation_name == BandOperationOptions.MINIMUM.name:
                result = band1.min(band2)
            elif operation_name == BandOperationOptions.MEAN.name:
                result = band1.add(band2).divide(2.0)
            else:
                raise ValueError(f"Unknown operation: {operation_name}")

            # Rename the result band
            result = result.rename(self.output_band_name)

            # Add the calculated band to the original image
            final_image = image.addBands(result)

            LOGGER.warning(
                f"Successfully calculated {self.output_band_name} using "
                f"{band1_name} {operation_name} {band2_name}"
            )

            return knut.export_gee_image_connection(final_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Band calculation failed: {e}")
            raise


############################################
# Image Get Info
############################################


@knext.node(
    name="GEE Image Info Extractor",
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
@knext.output_table(
    name="Image Info Table",
    description="Table containing image information as JSON for use in automated workflows",
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
        import pandas as pd

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

            # Create table with JSON cell for automated workflows
            df = pd.DataFrame([{"image_info_json": json_string}])

            LOGGER.warning("Successfully retrieved image information")
            return knext.Table.from_pandas(df), knext.view_html(html)

        except Exception as e:
            LOGGER.error(f"Failed to get image info: {e}")
            # Return error view and empty table
            error_html = f"""
            <div style="color: red; font-family: monospace;">
                <h3>Error</h3>
                <p>Failed to retrieve image information: {str(e)}</p>
            </div>
            """
            # Return empty table with error message
            df = pd.DataFrame([{"image_info_json": json.dumps({"error": str(e)})}])
            return knext.Table.from_pandas(df), knext.view_html(error_html)


############################################
# Image Clip
############################################


@knext.node(
    name="GEE Image Cliper",
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
    name="GEE Image Exporter",
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
        label="Destination mode",
        description="Choose between exporting to Google Drive or Google Cloud Storage.",
        default_value=DestinationModeOptions.get_default().name,
        enum=DestinationModeOptions,
    )

    drive_path = knext.StringParameter(
        "Drive path",
        "Drive folder and file name, e.g., 'GEEexport/my_image.tiff'.",
        default_value="GEEexport/export.tiff",
    ).rule(
        knext.OneOf(destination, [DestinationModeOptions.DRIVE.name]),
        knext.Effect.SHOW,
    )

    cloud_object_path = knext.StringParameter(
        "Cloud storage path",
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
        "Wait for download completion",
        "If enabled, the node will wait until the export finishes before completing.",
        default_value=True,
    )

    max_pixels = knext.IntParameter(
        "Max pixels",
        "Maximum number of pixels Earth Engine is allowed to read during export (integer ≤ 2,147,483,647).",
        default_value=1000000000,
        min_value=1,
        is_advanced=True,
    )

    max_wait_seconds = knext.IntParameter(
        "Max wait seconds",
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
# GeoTIFF To GEE Image
############################################


@knext.node(
    name="Cloud GeoTIFF to GEE Image",
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
        "Cloud storage GeoTIFF path",
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


############################################
# Image Value Filter
############################################


@knext.node(
    name="Image Value Filter",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageValueFilter.png",
    id="imagevaluefilter",
    after="bandselector",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="Filtered GEE Image connection.",
    port_type=gee_image_port_type,
)
class ImageValueFilter:
    """Filters an image by applying a single-band comparison.

    Choose the band, comparison operator, and threshold. Pixels that satisfy the condition are retained;
    others are masked (or optionally set to 0).
    """

    band_name = knext.StringParameter(
        "Band name",
        "Name of the band to evaluate (must exist in the image).",
        default_value="",
    )

    operator = knext.StringParameter(
        "Operator",
        "Comparison operator to apply.",
        default_value=">=",
        enum=[">=", ">", "<=", "<", "==", "!="],
    )

    threshold = knext.DoubleParameter(
        "Threshold",
        "Threshold value for comparison.",
        default_value=0.0,
    )

    retain_values = knext.BoolParameter(
        "Retain original values",
        "If enabled the original pixel value is kept; otherwise filtered pixels are set to 1.",
        default_value=False,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        image = image_connection.image

        band = (self.band_name or "").strip()
        if not band:
            raise ValueError("Band name is required for Image Value Filter.")

        try:
            image = image.select([band])
        except Exception as exc:
            raise ValueError(f"Band '{band}' not found in image: {exc}")

        threshold = ee.Number(self.threshold)

        ops = {
            ">=": image.gte,
            ">": image.gt,
            "<=": image.lte,
            "<": image.lt,
            "==": image.eq,
            "!=": image.neq,
        }

        mask_image = ops[self.operator](threshold)

        if self.retain_values:
            filtered_image = image_connection.image.updateMask(mask_image)
        else:
            filtered_image = mask_image.updateMask(mask_image)

        LOGGER.warning(
            "Applied value filter on band '%s' with operator %s %s",
            band,
            self.operator,
            self.threshold,
        )

        return knut.export_gee_image_connection(filtered_image, image_connection)


############################################
# Pixels To Feature Collection
############################################


@knext.node(
    name="Pixels to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "Pixel2FC.png",
    id="pixelstofc",
    after="imageclip",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded (binary) image object.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with extracted polygons.",
    port_type=gee_feature_collection_port_type,
)
class PixelsToFeatureCollection:
    """Vectorizes selected pixels and optionally merges them.

    The input image should be binary (mask or values 0/1). Regions where the band equals 1
    are converted into polygons. Optional unary union merges all polygons into a single MultiPolygon.
    """

    band_name = knext.StringParameter(
        "Band name",
        "Name of the band to vectorize. Available bands can be explored using the **GEE Image Info Extractor** node.",
        default_value="",
    )

    scale = knext.IntParameter(
        "Vectorization scale (meters)",
        "Resolution in meters to use when tracing polygons.",
        default_value=30,
        min_value=1,
        max_value=1000,
    )

    max_pixels = knext.IntParameter(
        "Max pixels",
        "Maximum number of pixels Earth Engine is allowed to process during vectorization (integer ≤ 2,147,483,647).",
        default_value=1000000000,
        min_value=1,
        is_advanced=True,
    )

    apply_union = knext.BoolParameter(
        "Apply unary union",
        "If enabled, the resulting polygons are dissolved into a single MultiPolygon feature.",
        default_value=True,
    )

    union_error_margin = knext.DoubleParameter(
        "Union error margin (meters)",
        "Error margin used when dissolving polygons (only when union is enabled).",
        default_value=1.0,
        min_value=0.1,
        is_advanced=True,
    ).rule(knext.OneOf(apply_union, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        image = image_connection.image
        band = (self.band_name or "").strip()
        if not band:
            raise ValueError("Band name is required for Pixels to Feature Collection.")

        try:
            single_band = image.select([band])
        except Exception as exc:
            raise ValueError(f"Band '{band}' not found in image: {exc}")

        mask = single_band.updateMask(single_band)

        LOGGER.warning(
            "Vectorizing band '%s' at scale %s m (maxPixels=%s, union=%s)",
            band,
            self.scale,
            self.max_pixels,
            self.apply_union,
        )

        vectors = mask.reduceToVectors(
            geometry=image.geometry(),
            scale=self.scale,
            maxPixels=self.max_pixels,
        )

        if self.apply_union:
            dissolved_geometry = vectors.geometry().dissolve(
                maxError=self.union_error_margin
            )
            dissolved = ee.FeatureCollection(
                ee.Feature(
                    dissolved_geometry,
                    {"source": "pixels_to_feature_collection"},
                )
            )
            output_fc = dissolved
        else:
            output_fc = vectors

        return knut.export_gee_feature_collection_connection(
            output_fc, image_connection
        )
