"""
GEE Image nodes for KNIME.
Single-image I/O and processing: create, read, merge, clip, mask, conditional assignment, export.
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
    name="GEE Image",
    description="Single-image I/O and processing: create, read, merge, clip, mask, conditional assignment, export.",
    icon="icons/ImageIO.png",
    after="imagecollection",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/image/"


############################################
# GEE Image Constant Creator
############################################


@knext.node(
    name="GEE Image Constant Creator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "constantImage.png",
    id="imageconstantcreator",
    after="",
)
@knext.input_port(
    name="Reference Image",
    description="Reference image to define the geometry. The constant image will be clipped to this image's geometry.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with constant value image clipped to reference geometry.",
    port_type=gee_image_port_type,
)
class ImageConstantCreator:
    """Creates a constant image with a specified value, clipped to a reference image geometry.

    This node creates a Google Earth Engine image where all pixels have the same constant value,
    automatically clipped to match the geometry of the reference image. This is more efficient than
    creating a global constant image and avoids the need for a separate clip operation.

    **Use Cases:**

    - Create base images for conditional operations (e.g., `.where()` operations)
    - Initialize classification images with default values
    - Create mask templates for further processing
    - Generate reference images for comparison

    **Common Use Case:**

    - Create a constant image with value 1, matching the geometry of a reference image (e.g., NDVI image)
    - Then use **Conditional Assignment** node to assign different values based on conditions
    - Initialize a classification image before applying classification rules

    """

    constant_value = knext.DoubleParameter(
        "Constant value",
        "The constant value for all pixels in the image.",
        default_value=1.0,
    )

    band_name = knext.StringParameter(
        "Band name",
        "Name for the output band of the constant image.",
        default_value="constant",
    )

    def configure(self, configure_context, reference_image_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        reference_image_connection,
    ):
        import ee

        # Get reference image and its geometry
        reference_image = reference_image_connection.image
        geometry = reference_image.geometry()

        # Create constant image and clip to reference geometry
        constant_image = ee.Image(self.constant_value).rename(self.band_name)
        constant_image = constant_image.clip(geometry)

        # Use reference_image_connection to get credentials and project_id
        return knut.export_gee_image_connection(
            constant_image, reference_image_connection
        )


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

    This node loads a single GEE image using Image name to select the dataset, and is commonly used as the entry point for single-image workflows.

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
            asset = ee.data.getAsset(imagename)
        except Exception as e:
            LOGGER.error(f"Failed to fetch asset metadata for '{imagename}': {e}")
            raise ValueError(
                f"Asset '{imagename}' was not found or is not accessible. "
                f"Please verify the image ID and your access permissions. Error: {str(e)}"
            ) from e

        asset_type = (asset or {}).get("type")
        if asset_type != "IMAGE":
            suggested = {
                "IMAGE_COLLECTION": "GEE Image Collection Reader",
                "FEATURE_COLLECTION": "GEE Feature Collection Reader",
            }.get(asset_type, "the appropriate reader node")
            raise ValueError(
                f"Asset '{imagename}' is not an Image (type: {asset_type}). "
                f"Please use {suggested}."
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

    This node selects bands using Bands and Use Regex parameters to control which bands are kept, and is commonly used to reduce data size or prepare inputs for band math.

    Available bands vary by image source and can be explored using the **GEE Image Info Extractor** node.

    **Common Band Combinations:**

    - **RGB**: 'B4,B3,B2' (Sentinel-2) or 'B4,B3,B2' (Landsat 8)
    - **False Color**: 'B8,B4,B3' (Sentinel-2) - good for vegetation
    - **SWIR**: 'B12,B8,B4' (Sentinel-2) - good for moisture detection
    - **NDVI Bands**: 'B8,B4' (Sentinel-2) - for vegetation index calculation


    **Note:** If no specified bands are found in the image, the original image is returned unchanged.

    **Use Regex:** When checked, the Bands field is interpreted as a single regular expression (Java regex syntax;
    matching is done on the GEE server). E.g. ``SR_B.`` or ``SR_B[0-9]`` to match Landsat SR band names.
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated band names (e.g. 'B1,B2,B3'), or a single regex when **Use Regex** is checked. "
        "Leave empty to keep all bands (when Use Regex is unchecked).",
        default_value="",
    )

    use_regex = knext.BoolParameter(
        "Use Regex",
        "Interpret Bands as a Java regex pattern (matched on the GEE server). Uncheck for comma-separated names.",
        default_value=False,
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
        image = image_connection.image
        bands_str = self.bands.strip()

        if self.use_regex:
            if not bands_str:
                LOGGER.warning(
                    "Use Regex is checked but Bands is empty; returning image unchanged."
                )
            else:
                image = image.select(bands_str)
        else:
            if bands_str:
                original_bands = image.bandNames().getInfo()
                band_list = [b.strip() for b in self.bands.split(",") if b.strip()]
                available_bands = [b for b in band_list if b in original_bands]
                if available_bands:
                    image = image.select(available_bands)
                else:
                    LOGGER.warning(
                        "No specified bands found in image. Available bands: %s",
                        original_bands,
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

    This node renames bands using Bands to rename and New band names parameters, and is commonly used to standardize band naming before merging or analysis.

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

    This node merges a Base Image and Additional Image using Overwrite existing bands to control duplicate handling, and is commonly used to stack bands from multiple sources.

    **Duplicate band names:**

    - **Overwrite existing bands = False (default):** Bands from the additional image that have the same
      name as a band in the base image are renamed with a numeric suffix (e.g. ``first`` → ``first_1``)
      so both are kept.
    - **Overwrite existing bands = True:** Bands from the additional image replace base bands with the
      same name.

    **Upstream band names:** For predictable merge results, use nodes that set explicit band names
    (e.g. **Feature Collection to Image** binary → band **mask**; **Distance to Feature Collection** → band **distance**).

    **Use Cases:**

    - Merge distance and mask (e.g. Distance to FC + FC to Image) for Band Calculator
    - Combine multi-temporal data (e.g., nighttime lights from different years)
    - Stack bands from different sources; chain multiple Band Merger nodes for 3+ images
    """

    overwrite = knext.BoolParameter(
        "Overwrite existing bands",
        "If True, overwrite base bands with the same name. If False, additional bands that duplicate base names are renamed with a suffix (e.g. first_1) so both are kept.",
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
            base_bands = base_image.bandNames().getInfo()
            additional_image = additional_image_connection.image
            additional_bands = additional_image.bandNames().getInfo()
            base_set = set(base_bands)

            if not self.overwrite and base_set:
                # Rename duplicate additional bands with suffix so both base and additional are kept
                used = set(base_set)
                new_names = []
                for b in additional_bands:
                    if b in base_set:
                        suffix = 1
                        while f"{b}_{suffix}" in used:
                            suffix += 1
                        new_name = f"{b}_{suffix}"
                        used.add(new_name)
                        new_names.append(new_name)
                    else:
                        new_names.append(b)
                        used.add(b)
                additional_image = additional_image.select(additional_bands).rename(
                    new_names
                )

            merged_image = base_image.addBands(
                additional_image, overwrite=self.overwrite
            )

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
    name="GEE Image Info View",
    description="HTML view showing detailed image information",
)
class ImageGetInfo:
    """Displays detailed information about a Google Earth Engine image.

    This node outputs detailed metadata using Add nominal scale to include per-band resolution, and is commonly used to inspect band names and properties before processing.

    **Information Displayed:**

    - **Band Information**: Names, types, and properties of all bands
    - **Nominal Scale** (optional): Pixel resolution in meters for each band (enabled via parameter)
    - **Image Properties**: Metadata, acquisition date, cloud cover, etc.
    - **Geometry**: Bounding box and projection information
    - **System Properties**: GEE internal properties and identifiers

    **Note on Nominal Scale:**

    - Nominal scale extraction is **optional** and disabled by default (see **Add nominal scale** parameter)
    - When enabled, each band's **nominal_scale** is extracted and added to the band information
    - Different bands may have different resolutions (e.g., Sentinel-2: 10m, 20m, 60m bands)
    - The nominal_scale is the pixel resolution in meters, useful for determining appropriate sampling scales
    - **Performance**: Extracting nominal scale requires a separate API call per band, which can be slow for images with many bands

    **Use Cases:**

    - Explore image structure and available bands
    - Verify image properties and band resolutions before analysis
    - Debug image processing issues
    - Understand image metadata and acquisition parameters
    - Determine appropriate scale for sampling operations
    """

    add_nominal_scale = knext.BoolParameter(
        "Add nominal scale",
        """If enabled, extracts and adds nominal_scale (pixel resolution in meters) for each band.
        
        **Performance Note:**
        - This requires a separate GEE API call for each band
        - For images with many bands (e.g., Sentinel-2 with 20+ bands), this can be slow

        - The nominal_scale is useful for determining appropriate sampling scales in other nodes
        """,
        default_value=False,
        is_advanced=True,
    )

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

            # Add nominal_scale for each band (optional, can be slow for many bands)
            # Different bands may have different resolutions (e.g., Sentinel-2: 10m, 20m, 60m)
            if (
                self.add_nominal_scale
                and "bands" in info
                and isinstance(info["bands"], list)
            ):
                LOGGER.warning(
                    f"Extracting nominal scale for {len(info['bands'])} bands..."
                )
                for band in info["bands"]:
                    band_id = band.get("id")
                    if band_id:
                        try:
                            # Get nominal scale for this specific band
                            nominal_scale = (
                                image.select(band_id)
                                .projection()
                                .nominalScale()
                                .getInfo()
                            )
                            band["nominal_scale"] = nominal_scale
                            LOGGER.warning(
                                f"Band {band_id}: nominal_scale = {nominal_scale} meters"
                            )
                        except Exception as e:
                            LOGGER.warning(
                                f"Could not get nominalScale for band {band_id}: {e}"
                            )
                            band["nominal_scale"] = None
                LOGGER.warning("Successfully added nominal_scale for all bands")

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
# Image Geometry to Feature Collection
############################################


@knext.node(
    name="GEE Image Boundary to Geometry",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "imageBoundary.png",
    id="imagegeometrytofc",
    after="imageclip",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection to extract geometry from.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection containing the image geometry as a single feature.",
    port_type=gee_feature_collection_port_type,
)
class ImageGeometry:
    """Extracts the geometry from an image and creates a Feature Collection.

    This node extracts image bounds into a Feature Collection with no parameters and is commonly used to create an ROI that matches a reference image.

    **Use Cases:**

    - Convert image geometry to Feature Collection for use with Image Clip node
    - Create ROI boundaries from image extents
    - Use image boundaries in spatial filtering operations
    - Prepare geometry for other Feature Collection operations

    **Example Workflow:**

    - Extract geometry from reference image: Image → This node → Feature Collection
    - Use for clipping: Constant Image → Image Clip (using FC from this node) → Clipped Image

    """

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

        try:
            # Extract geometry from image
            geometry = image.geometry()

            # Create a single feature with the geometry
            feature = ee.Feature(geometry, {"source": "image_geometry"})

            # Create Feature Collection with single feature
            feature_collection = ee.FeatureCollection([feature])

            LOGGER.warning(
                "Successfully extracted image geometry to Feature Collection"
            )
            return knut.export_gee_feature_collection_connection(
                feature_collection, image_connection
            )

        except Exception as e:
            LOGGER.error(f"Failed to extract image geometry: {e}")
            raise


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

    This node exports an image using Destination mode, path, scale, and wait options to control where and how the image is written, and is commonly used to deliver results outside GEE.

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

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        scale_description="The scale in meters per pixel for export (lower = higher resolution). Only used when Use NominalScale is disabled.",
    )

    wait_for_completion = knext.BoolParameter(
        "Wait for download completion",
        "If enabled, the node will wait until the export finishes before completing.",
        default_value=True,
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=knut.GEE_MAX_PIXELS,
        min_value=1,
        description="Maximum number of pixels Earth Engine is allowed to read during export.",
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
        scale_value = knut.resolve_scale(self.use_nominal_scale, self.scale, image)
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
            scale_value,
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
                    scale=scale_value,
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
            scale=scale_value,
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
    name="GEE Cloud GeoTIFF to Image",
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
    name="GEE Image Info View",
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
    name="GEE Image Value Filter",
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
    __doc__ = (
        "Filters an image by applying a single band comparison.\n\n"
        "This node filters pixels using Band name, Operator, and Value parameters plus output mode switches, and is commonly used to create masks or thresholded images.\n\n"
        + knut.IMAGE_VALUE_FILTER_FORMULA
        + "\n\n**Output modes** (controlled by Retain original values and Set false pixels to 0):\n\n"
        "• **Mode A** (Retain ✓): Keep original pixel values; pixels not satisfying the condition are masked (transparent). "
        "Use for visualizing continuous data (e.g. NDVI) only where condition holds.\n\n"
        "• **Mode B** (Retain ✗, Set false to 0 ✓): Output 0/1 binary image; both 0 and 1 are visible. "
        "Use to create binary masks for downstream nodes (e.g. mode filter, Mask Apply).\n\n"
        "• **Mode C** (Retain ✗, Set false to 0 ✗): Output 0/1 binary; only pixels with value 1 are visible, 0 are masked. "
        "Use to display only the matching region (e.g. mask out non-forest from a thresholded image).\n\n"
        "• **Mode D** (Retain ✓, Set false to 0 ✓): Keep original values; non-matching pixels are set to 0 (visible). "
        "Use when you need continuous values and 0 for non-matching (no masking)."
    )

    band_name = knext.StringParameter(
        "Band name",
        "Name of the band to evaluate (must exist in the image).",
        default_value="",
    )

    operator = knext.StringParameter(
        "Operator",
        "Comparison: >=, >, <=, <, ==, != . Pixels satisfying are kept.",
        default_value=">=",
        enum=[">=", ">", "<=", "<", "==", "!="],
    )

    threshold = knext.DoubleParameter(
        "Value",
        "Value to compare the band against.",
        default_value=0.0,
    )

    retain_values = knext.BoolParameter(
        "Retain original values",
        "✓ = Mode A/D: Keep original values (A=mask non-matching, D=set to 0). ✗ = Modes B/C: Output binary 1/0.",
        default_value=False,
    )

    set_false_to_zero = knext.BoolParameter(
        "Set false pixels to 0",
        "✓ = Mode B/D: Non-matching pixels become 0 (visible). ✗ = Mode A/C: Non-matching pixels are masked.",
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

        if self.retain_values and self.set_false_to_zero:
            # Mode D: original where true, 0 where false (both visible)
            base = image_connection.image
            filtered_image = base.where(mask_image.eq(0), base.multiply(0))
        elif self.retain_values:
            # Mode A: original values, non-matching masked
            filtered_image = image_connection.image.updateMask(mask_image)
        elif self.set_false_to_zero:
            # Mode B: binary 0/1, both visible
            filtered_image = mask_image
        else:
            # Mode C: binary 0/1, only 1 visible
            filtered_image = mask_image.updateMask(mask_image)

        LOGGER.warning(
            "Applied value filter on band '%s' with operator %s %s",
            band,
            self.operator,
            self.threshold,
        )

        return knut.export_gee_image_connection(filtered_image, image_connection)


############################################
# Image Mask Apply
############################################


@knext.node(
    name="GEE Image Mask Apply",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "imageMask.png",
    id="imagemaskapply",
    after="imagemultibandcalculator",
)
@knext.input_port(
    name="Base Image",
    description="The image to apply the mask to.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Mask Image",
    description="The mask image. Non-zero pixels in the mask will be retained; zero pixels will be masked out.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with mask applied.",
    port_type=gee_image_port_type,
)
class ImageMaskApply:
    __doc__ = (
        "Applies a mask to an image using updateMask: pixels where mask is 0 are masked out and non-zero retained.\n\n"
        "This node applies a mask using Mask band name to select the mask band, and is commonly used to apply QA masks or binary masks to an image.\n\n"
        "**Operation:** Base image × Mask image (one band); result = base with pixels masked where mask band == 0.\n\n"
        "**Common workflow:** " + knut.MASK_APPLY_WORKFLOW_EXAMPLE
    )

    mask_band = knext.StringParameter(
        "Mask band name",
        "Name of the band from mask image to use as mask. Leave empty to use the first band if mask image has multiple bands.",
        default_value="",
    )

    def configure(self, configure_context, base_image_schema, mask_image_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        base_image_connection,
        mask_image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get images from connections
        base_image = base_image_connection.image
        mask_image = mask_image_connection.image

        try:
            # Get available bands from mask image
            available_bands = mask_image.bandNames().getInfo()

            # Select mask band
            mask_band = (self.mask_band or "").strip()
            if mask_band:
                # User specified a band
                if mask_band not in available_bands:
                    raise ValueError(
                        f"Mask band '{mask_band}' not found in mask image. Available bands: {available_bands}"
                    )
                mask_band_image = mask_image.select(mask_band)
            else:
                # Use first band if no band specified
                if len(available_bands) > 1:
                    LOGGER.warning(
                        f"Mask image has multiple bands {available_bands}. Using first band '{available_bands[0]}'. "
                        "Specify 'Mask band name' to select a different band."
                    )
                mask_band_image = mask_image.select(available_bands[0])

            # Apply mask: pixels where mask is non-zero are retained, zero pixels are masked out
            # In GEE, updateMask expects non-zero values to be kept, zero values to be masked
            masked_image = base_image.updateMask(mask_band_image)

            LOGGER.warning(
                f"Applied mask to base image using band '{mask_band or available_bands[0]}' from mask image"
            )

            return knut.export_gee_image_connection(masked_image, base_image_connection)

        except Exception as e:
            LOGGER.error(f"Mask application failed: {e}")
            raise


############################################
# Image Conditional Assignment (Where)
############################################


class ConditionalOperatorOptions(knext.EnumParameterOptions):
    """Options for conditional comparison operators."""

    GTE = (">=", "Greater than or equal to (≥)")
    GT = (">", "Greater than (>)")
    LTE = ("<=", "Less than or equal to (≤)")
    LT = ("<", "Less than (<)")
    EQ = ("==", "Equal to (==)")
    NEQ = ("!=", "Not equal to (!=)")

    @classmethod
    def get_default(cls):
        return cls.GTE


@knext.node(
    name="GEE Image Conditional Assignment",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "imageWhere.png",
    id="imageconditionalassignment",
    after="imagevaluefilter",
)
@knext.input_port(
    name="Base Image",
    description="The image to modify (where conditions will be applied).",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Condition Image",
    description="The image used to generate conditions (e.g., NDVI image for threshold comparisons).",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with conditionally assigned values.",
    port_type=gee_image_port_type,
)
class ImageConditionalAssignment:
    """Applies conditional value assignment using where operation.

    This node assigns values using Condition band name, Operator, Threshold, and Replacement value parameters, and is commonly used for rule-based classification or re-coding.

    **Operation:**

    - **Base Image**: The image to modify (where assignments will be applied)
    - **Condition Image**: The image used to generate boolean conditions
    - **Condition**: Band name from condition image + operator + threshold
    - **Replacement Value**: The value to assign when condition is true

    **Use Cases:**

    - Classify pixels based on thresholds (e.g., NDVI-based classification)
    - Create multi-class maps from continuous data
    - Apply conditional rules for land cover mapping
    - Replace specific pixel values based on conditions

    **Common Use Case:**

    - Create classification image: Start with constant image (value=1), then apply multiple
      `.where()` operations to assign different class values based on conditions
    - Water/Non-water classification: Where NDWI > threshold → set to 1, else 0
    """

    condition_band = knext.StringParameter(
        "Condition band name",
        "Name of the band from condition image to use for comparison (e.g., 'ndvi', 'B4').",
        default_value="",
    )

    operator = knext.EnumParameter(
        label="Operator",
        description="Comparison operator for the condition",
        default_value=ConditionalOperatorOptions.get_default().name,
        enum=ConditionalOperatorOptions,
    )

    threshold = knext.DoubleParameter(
        "Threshold",
        "Threshold value for comparison.",
        default_value=0.0,
    )

    replacement_value = knext.DoubleParameter(
        "Replacement value",
        "Value to assign to base image pixels where condition is true.",
        default_value=0.0,
    )

    def configure(self, configure_context, base_image_schema, condition_image_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        base_image_connection,
        condition_image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get images from connections
        base_image = base_image_connection.image
        condition_image = condition_image_connection.image

        # Validate condition band
        condition_band = (self.condition_band or "").strip()
        if not condition_band:
            raise ValueError(
                "Condition band name is required for Conditional Assignment."
            )

        try:
            # Get available bands from condition image
            available_bands = condition_image.bandNames().getInfo()
            if condition_band not in available_bands:
                raise ValueError(
                    f"Band '{condition_band}' not found in condition image. Available bands: {available_bands}"
                )

            # Select the condition band
            condition_band_image = condition_image.select(condition_band)

            # Create condition based on operator
            operator_name = self.operator
            threshold_ee = ee.Number(self.threshold)
            replacement_value_ee = ee.Number(self.replacement_value)

            # Build condition image (boolean) and get operator symbol for logging
            operator_symbol = None
            if operator_name == ConditionalOperatorOptions.GTE.name:
                condition = condition_band_image.gte(threshold_ee)
                operator_symbol = ">="
            elif operator_name == ConditionalOperatorOptions.GT.name:
                condition = condition_band_image.gt(threshold_ee)
                operator_symbol = ">"
            elif operator_name == ConditionalOperatorOptions.LTE.name:
                condition = condition_band_image.lte(threshold_ee)
                operator_symbol = "<="
            elif operator_name == ConditionalOperatorOptions.LT.name:
                condition = condition_band_image.lt(threshold_ee)
                operator_symbol = "<"
            elif operator_name == ConditionalOperatorOptions.EQ.name:
                condition = condition_band_image.eq(threshold_ee)
                operator_symbol = "=="
            elif operator_name == ConditionalOperatorOptions.NEQ.name:
                condition = condition_band_image.neq(threshold_ee)
                operator_symbol = "!="
            else:
                raise ValueError(f"Unknown operator: {operator_name}")

            # Apply .where() operation: base_image.where(condition, replacement_value)
            result_image = base_image.where(condition, replacement_value_ee)

            LOGGER.warning(
                f"Applied conditional assignment: where {condition_band} {operator_symbol} {self.threshold}, set to {self.replacement_value}"
            )

            return knut.export_gee_image_connection(result_image, base_image_connection)

        except Exception as e:
            LOGGER.error(f"Conditional assignment failed: {e}")
            raise


############################################
# Pixels To Feature Collection (Image to Feature Collection)
############################################


@knext.node(
    name="GEE Pixels to Feature Collection",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "Pixel2FC.png",
    id="pixelstofc",
    after="imageclip",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection (binary mask or categorical band to vectorize).",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with extracted polygons (and optional class property).",
    port_type=gee_feature_collection_port_type,
)
class PixelsToFeatureCollection:
    """Vectorizes image pixels to polygons. Supports binary masks or categorical bands.

    This node vectorizes pixels using Band name, Mode, and scale or union options to control polygon creation, and is commonly used to convert masks or classes into vector features.

    **Binary mode:** Input band is treated as 0/1 mask; non-zero pixels become polygons. Optional unary
    union merges all into one MultiPolygon (e.g. for single-region extraction).

    **Categorical mode:** Each distinct pixel value becomes a separate polygon; output features get a
    property (label) with the class value (e.g. elevation zones, forest loss year). Optional focalMode
    smooths the raster before vectorizing to reduce small isolated polygons.

    **Shared options:** Scale, maxPixels, eight-connected, and tile scale (use 2–4 if you
    get timeouts on large or high-resolution rasters). Output geometry is always polygon.
    """

    band_name = knext.StringParameter(
        "Band name",
        "Name of the band to vectorize. Available bands can be explored using the **GEE Image Info Extractor** node.",
        default_value="",
    )

    vectorize_mode = knext.StringParameter(
        "Mode",
        "Binary: 0/1 mask, one region or union. Categorical: preserve class values per polygon (e.g. zones, loss year).",
        default_value="binary",
        enum=["binary", "categorical"],
    )

    label_property = knext.StringParameter(
        "Label property name",
        "Feature property name for the pixel (class) value in categorical mode.",
        default_value="label",
    ).rule(knext.OneOf(vectorize_mode, ["categorical"]), knext.Effect.SHOW)

    use_focal_mode = knext.BoolParameter(
        "Smooth with focalMode before vectorizing",
        "Apply focalMode to reduce small isolated polygons (categorical mode). Uses kernel radius in pixels.",
        default_value=False,
    ).rule(knext.OneOf(vectorize_mode, ["categorical"]), knext.Effect.SHOW)

    focal_mode_radius = knext.IntParameter(
        "FocalMode kernel radius (pixels)",
        "Kernel radius for focalMode smoothing (e.g. 4 for 9x9 neighborhood).",
        default_value=4,
        min_value=1,
        max_value=20,
        is_advanced=True,
    ).rule(
        knext.And(
            knext.OneOf(vectorize_mode, ["categorical"]),
            knext.OneOf(use_focal_mode, [True]),
        ),
        knext.Effect.SHOW,
    )

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        max_value=1000,
        scale_description="Resolution in meters to use when tracing polygons. Only used when Use NominalScale is disabled.",
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=knut.GEE_MAX_PIXELS,
        min_value=1,
        description="Maximum number of pixels Earth Engine is allowed to process during vectorization.",
    )

    tile_scale = knext.IntParameter(
        "Tile scale",
        "Scaling factor for aggregation tile size (1–16). Use 2 or 4 if you get timeouts or memory errors on large or high-resolution rasters.",
        default_value=1,
        min_value=1,
        max_value=16,
        is_advanced=True,
    )

    apply_union = knext.BoolParameter(
        "Apply unary union",
        "If enabled, the resulting polygons are dissolved into a single MultiPolygon feature (binary mode only).",
        default_value=True,
    ).rule(knext.OneOf(vectorize_mode, ["binary"]), knext.Effect.SHOW)

    apply_union_per_class = knext.BoolParameter(
        "Apply union per class",
        "If enabled, polygons with the same label are dissolved into one MultiPolygon per class (fewer, larger features).",
        default_value=False,
    ).rule(knext.OneOf(vectorize_mode, ["categorical"]), knext.Effect.SHOW)

    union_error_margin = knext.DoubleParameter(
        "Union error margin (meters)",
        "Error margin used when dissolving polygons (binary unary union or categorical union per class).",
        default_value=1.0,
        min_value=0.1,
        is_advanced=True,
    ).rule(
        knext.Or(
            knext.And(
                knext.OneOf(vectorize_mode, ["binary"]),
                knext.OneOf(apply_union, [True]),
            ),
            knext.And(
                knext.OneOf(vectorize_mode, ["categorical"]),
                knext.OneOf(apply_union_per_class, [True]),
            ),
        ),
        knext.Effect.SHOW,
    )

    eight_connected = knext.BoolParameter(
        "Eight-connected",
        "Use 8-connectedness for grouping pixels (default 4-connected).",
        default_value=False,
        is_advanced=True,
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
            raise ValueError("Band name is required for Pixels to Feature Collection.")

        try:
            single_band = image.select([band])
        except Exception as exc:
            raise ValueError(f"Band '{band}' not found in image: {exc}")

        geom = image.geometry()
        scale = knut.resolve_scale(self.use_nominal_scale, self.scale, image)
        max_pixels = self.max_pixels

        if self.vectorize_mode == "binary":
            mask = single_band.updateMask(single_band)
            LOGGER.warning(
                "Vectorizing band '%s' (binary) at scale %s m (maxPixels=%s, union=%s)",
                band,
                scale,
                max_pixels,
                self.apply_union,
            )
            vectors = mask.reduceToVectors(
                geometry=geom,
                scale=scale,
                maxPixels=max_pixels,
                geometryType="polygon",
                eightConnected=self.eight_connected,
                tileScale=self.tile_scale,
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
        else:
            # Categorical: optionally smooth then reduceToVectors with label
            to_vectorize = single_band
            if self.use_focal_mode:
                to_vectorize = single_band.focalMode(
                    radius=self.focal_mode_radius,
                    kernelType="square",
                )
                to_vectorize = to_vectorize.reproject(single_band.projection())
                LOGGER.warning(
                    "Applied focalMode (radius=%s) before vectorizing.",
                    self.focal_mode_radius,
                )
            LOGGER.warning(
                "Vectorizing band '%s' (categorical) at scale %s m (labelProperty=%s, unionPerClass=%s)",
                band,
                scale,
                self.label_property,
                self.apply_union_per_class,
            )
            vectors = to_vectorize.reduceToVectors(
                geometry=geom,
                scale=scale,
                maxPixels=max_pixels,
                geometryType="polygon",
                labelProperty=self.label_property,
                eightConnected=self.eight_connected,
                tileScale=self.tile_scale,
            )
            if self.apply_union_per_class:
                # One feature per class: dissolve all polygons with same label into one MultiPolygon
                label_prop = self.label_property
                err_margin = self.union_error_margin
                distinct_labels = vectors.aggregate_array(label_prop).distinct()

                def make_feature(label):
                    sub = vectors.filter(ee.Filter.eq(label_prop, label))
                    dissolved = sub.geometry().dissolve(maxError=err_margin)
                    return ee.Feature(dissolved, {label_prop: label})

                output_fc = ee.FeatureCollection(distinct_labels.map(make_feature))
            else:
                output_fc = vectors

        return knut.export_gee_feature_collection_connection(
            output_fc, image_connection
        )
