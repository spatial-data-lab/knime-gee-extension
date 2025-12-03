import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
    gee_feature_collection_port_type,
    gee_image_collection_port_type,
)


__category = knext.category(
    path="/community/gee",
    level_id="visualize",
    name="GEE View",
    description="Visualize nodes for Google Earth Engine",
    icon="icons/visualize.png",
    after="featureio",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/visualize/"


############################################
# GEEMap View Node
############################################
@knext.node(
    name="GEE Image View",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=__NODE_ICON_PATH + "imageView.png",
    category=__category,
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
@knext.output_view(
    name="GEE Map View",
    description="Showing a map with the GEE map",
    static_resources="libs/leaflet/1.9.3",
)
class ViewNodeGEEMap:
    """Visualizes a GEE Image on a map.

    This node will visualize the given GEE Image on a map using the [geemap](https://geemap.org/) library.
    This view is highly interactive and allows you to change various aspects of the view within the visualization itself.
    This node provides two visualization modes for GEE Images:

    1. **Single Band Mode**: For images with 1-2 bands, displays the first band with color mapping
    2. **RGB Mode**: For images with 3+ bands, automatically combines the first 3 bands as RGB

    **Visualization Features**:
    - **Auto Statistics**: Automatically calculates min/max values from non-zero pixels for optimal visualization

    - **Manual Control**: Override auto statistics with custom min/max values

    - **Color Palettes**: Support for custom color gradients using comma-separated hex codes

    - **Transparency**: Adjustable alpha channel for overlay visualization

    - **Base Maps**: Choose from OpenStreetMap, Satellite, Terrain, or Hybrid backgrounds

    **Usage**:
    - For single-band data: The node automatically uses the first band with color mapping

    - For multi-band data: The node automatically uses the first 3 bands as RGB channels

    - Enable "Auto Statistics" to let the node calculate optimal visualization ranges

    - Disable "Auto Statistics" to manually set min/max values for precise control

    """

    color_palette = knext.StringParameter(
        "Color palette",
        """Comma-separated color codes for gradient. Supports hex codes with or without # prefix.
                
        Common gradient combinations:
        • Terrain: 000080,00FFFF,00FF00,FFFF00,FF0000 (blue-cyan-green-yellow-red)

        • Heatmap: 0000FF,00FFFF,00FF00,FFFF00,FF0000 (blue-cyan-green-yellow-red)

        • Vegetation: 8B4513,FFFF00,00FF00 (brown-yellow-green)

        • Elevation: 000080,00FF00,FFFF00,FF8000,FF0000 (blue-green-yellow-orange-red)

        • Temperature: 0000FF,00FFFF,00FF00,FFFF00,FF0000 (cold to hot)
        
        • Simple: 000000,FFFFFF (black to white)

        Examples: '000080,00FF00,FF0000' or '#000080,#00FF00,#FF0000'""",
        default_value="000000,FFFFFF",
    )

    alpha = knext.DoubleParameter(
        "Alpha (Transparency)",
        "Transparency level (0.0 = fully transparent, 1.0 = fully opaque)",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
    )
    # Single band parameters (for manual override)
    auto_stats = knext.BoolParameter(
        "Auto statistics",
        "Automatically calculate min/max values from non-zero pixels",
        default_value=True,
        is_advanced=True,
    )

    min_value = knext.DoubleParameter(
        "Minimum value",
        "Minimum value for color mapping (ignored if auto statistics is enabled)",
        default_value=0.0,
        is_advanced=True,
    ).rule(knext.OneOf(auto_stats, [False]), knext.Effect.SHOW)

    max_value = knext.DoubleParameter(
        "Maximum value",
        "Maximum value for color mapping (ignored if auto statistics is enabled)",
        default_value=0.3,
        is_advanced=True,
    ).rule(knext.OneOf(auto_stats, [False]), knext.Effect.SHOW)

    # Base map options
    base_map = knext.StringParameter(
        "Base map",
        "Base map layer for visualization",
        default_value="OpenStreetMap",
        enum=["OpenStreetMap", "Satellite", "Terrain", "Hybrid"],
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import geemap.foliumap as geemap
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image directly from connection object
        # No need to initialize GEE - it's already initialized in the same Python process!
        image = image_connection.image

        # Get band names efficiently without full image info
        band_names = image.bandNames().getInfo()

        # Auto-detect display mode based on number of bands
        if len(band_names) >= 3:
            # RGB mode - automatically use first 3 bands
            red_band = band_names[0]
            green_band = band_names[1]
            blue_band = band_names[2]

            # Calculate statistics if auto_stats is enabled (for RGB mode)
            if self.auto_stats:
                try:
                    # Get image statistics for all three RGB bands
                    rgb_bands = [red_band, green_band, blue_band]
                    stats = (
                        image.select(rgb_bands)
                        .reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,  # Use 1km scale for efficiency
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    # Get min and max across all three bands
                    min_vals = [stats.get(f"{band}_min", 0) for band in rgb_bands]
                    max_vals = [stats.get(f"{band}_max", 0) for band in rgb_bands]

                    # Use the overall min and max across all bands
                    min_val = min([v for v in min_vals if v is not None])
                    max_val = max([v for v in max_vals if v is not None])

                    # Filter out zero values for better visualization
                    if min_val == 0:
                        # Get non-zero minimum across all bands
                        non_zero_image = (
                            image.select(rgb_bands)
                            .gt(0)
                            .multiply(image.select(rgb_bands))
                        )
                        non_zero_stats = non_zero_image.reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        ).getInfo()

                        non_zero_min_vals = [
                            non_zero_stats.get(f"{band}_min", min_val)
                            for band in rgb_bands
                        ]
                        min_val = min(
                            [v for v in non_zero_min_vals if v is not None and v > 0]
                            or [min_val]
                        )

                    # Ensure max is greater than min
                    if max_val <= min_val:
                        max_val = min_val + 1 if min_val is not None else 1

                except Exception as e:
                    # LOGGER.warning(
                    #     f"Failed to calculate auto stats for RGB: {e}, using manual values"
                    # )
                    min_val = self.min_value
                    max_val = self.max_value
            else:
                min_val = self.min_value
                max_val = self.max_value

            vis = {
                "bands": [red_band, green_band, blue_band],
                "min": min_val,
                "max": max_val,
                "opacity": self.alpha,
            }
            # LOGGER.warning(f"RGB mode: {red_band}, {green_band}, {blue_band}, range: {min_val}-{max_val}")

        else:
            # Single band mode - use first available band
            single_band = band_names[0] if band_names else "B1"

            # Parse color palette and ensure all colors have # prefix
            color_list = []
            for color in self.color_palette.split(","):
                color = color.strip()
                # Add # prefix if not present
                if not color.startswith("#"):
                    color = "#" + color
                color_list.append(color)

            # Calculate statistics if auto_stats is enabled
            if self.auto_stats:
                try:
                    # Get image statistics for the single band
                    stats = (
                        image.select(single_band)
                        .reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,  # Use 1km scale for efficiency
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    min_val = stats.get(f"{single_band}_min", 0)
                    max_val = stats.get(f"{single_band}_max", 1)

                    # Filter out zero values for better visualization
                    if min_val == 0:
                        # Get non-zero minimum
                        non_zero_stats = (
                            image.select(single_band)
                            .gt(0)
                            .multiply(image.select(single_band))
                            .reduceRegion(
                                reducer=ee.Reducer.minMax(),
                                geometry=image.geometry(),
                                scale=1000,
                                maxPixels=1e9,
                            )
                            .getInfo()
                        )

                        min_val = non_zero_stats.get(f"{single_band}_min", min_val)
                        if min_val == 0:  # If still zero, use original min
                            min_val = stats.get(f"{single_band}_min", 0)

                    # LOGGER.warning(
                    #     f"Auto-calculated stats for {single_band}: min={min_val}, max={max_val}"
                    # )

                except Exception as e:
                    # LOGGER.warning(
                    #     f"Failed to calculate auto stats: {e}, using manual values"
                    # )
                    min_val = self.min_value
                    max_val = self.max_value
            else:
                min_val = self.min_value
                max_val = self.max_value

            vis = {
                "bands": [single_band],
                "min": min_val,
                "max": max_val,
                "palette": color_list,
                "opacity": self.alpha,
            }
            # LOGGER.warning(
            #     f"Single band mode: {single_band}, colors: {color_list}, range: {min_val}-{max_val}"
            # )

        # Create map
        Map = geemap.Map()

        # Add base map layer
        if self.base_map == "OpenStreetMap":
            Map.add_basemap("OpenStreetMap")
        elif self.base_map == "Satellite":
            Map.add_basemap("Satellite")
        elif self.base_map == "Terrain":
            Map.add_basemap("Terrain")
        elif self.base_map == "Hybrid":
            Map.add_basemap("Hybrid")

        Map.addLayer(image, vis, "GEE Image")

        # center the map automatically based on image bounds
        Map.centerObject(image)
        # replace css and JavaScript paths
        html = Map.get_root().render()

        return knext.view(html)


############################################
# GEE Image Collection View Node
############################################
@knext.node(
    name="GEE Image Collection View",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=__NODE_ICON_PATH + "collectionView.png",
    category=__category,
    after="",
)
@knext.input_port(
    name="GEE Image Collection Connection",
    description="GEE Image Collection connection with embedded collection object.",
    port_type=gee_image_collection_port_type,
)
@knext.output_view(
    name="GEE Map View",
    description="Showing a map with the GEE Image Collection",
    static_resources="libs/leaflet/1.9.3",
)
class ViewNodeGEEImageCollection:
    """Visualizes a GEE Image Collection on a map.

    This node will visualize the given GEE Image Collection on a map using the [geemap](https://geemap.org/) library.
    The collection is automatically mosaicked (most recent pixel shown) before visualization.
    This view is highly interactive and allows you to change various aspects of the view within the visualization itself.

    **Image Collection Handling:**
    - Automatically creates a mosaic from the collection (most recent pixel shown)
    - Equivalent to GEE JavaScript: Map.addLayer(collection, {bands: ['B4','B3','B2'], ...})

    **Band Selection:**
    - Specify which bands to visualize (e.g., 'B4,B3,B2' for RGB)
    - Leave empty to auto-use first 3 bands for RGB or first band for single-band
    - Common RGB combinations:
      - Landsat: 'B4,B3,B2' (True Color)
      - Sentinel-2: 'B4,B3,B2' (True Color)
      - False Color: 'B8,B4,B3' (Sentinel-2 vegetation)

    This node provides two visualization modes:

    1. **Single Band Mode**: For 1-2 selected bands, displays the first band with color mapping
    2. **RGB Mode**: For 3+ selected bands, uses first 3 bands as RGB channels

    **Visualization Features**:
    - **Auto Statistics**: Automatically calculates min/max values from non-zero pixels for optimal visualization
    - **Manual Control**: Override auto statistics with custom min/max values
    - **Color Palettes**: Support for custom color gradients using comma-separated hex codes
    - **Transparency**: Adjustable alpha channel for overlay visualization
    - **Base Maps**: Choose from OpenStreetMap, Satellite, Terrain, or Hybrid backgrounds

    **Usage**:
    - Specify bands to visualize (e.g., 'B4,B3,B2' for RGB visualization)
    - Leave bands empty to auto-use first available bands
    - Enable "Auto Statistics" to let the node calculate optimal visualization ranges
    - Disable "Auto Statistics" to manually set min/max values for precise control

    """

    bands = knext.StringParameter(
        "Bands to visualize",
        """Comma-separated list of band names to visualize (e.g., 'B4,B3,B2').
        For RGB visualization, provide 3 bands (Red, Green, Blue order).
        For single-band visualization, provide 1 band.
        Leave empty to auto-use first 3 bands (RGB) or first band (single-band).
        
        Common combinations:
        - Landsat True Color: 'B4,B3,B2'
        - Sentinel-2 True Color: 'B4,B3,B2'
        - False Color Vegetation: 'B8,B4,B3' (Sentinel-2)
        - SWIR: 'B12,B8,B4' (Sentinel-2)""",
        default_value="",
    )

    color_palette = knext.StringParameter(
        "Color palette",
        """Comma-separated color codes for gradient. Supports hex codes with or without # prefix.
        Only used for single-band visualization.
                
        Common gradient combinations:
        • Terrain: 000080,00FFFF,00FF00,FFFF00,FF0000 (blue-cyan-green-yellow-red)
        • Heatmap: 0000FF,00FFFF,00FF00,FFFF00,FF0000 (blue-cyan-green-yellow-red)
        • Vegetation: 8B4513,FFFF00,00FF00 (brown-yellow-green)
        • Elevation: 000080,00FF00,FFFF00,FF8000,FF0000 (blue-green-yellow-orange-red)
        • Temperature: 0000FF,00FFFF,00FF00,FFFF00,FF0000 (cold to hot)
        • Simple: 000000,FFFFFF (black to white)

        Examples: '000080,00FF00,FF0000' or '#000080,#00FF00,#FF0000'""",
        default_value="000000,FFFFFF",
    )

    alpha = knext.DoubleParameter(
        "Alpha (Transparency)",
        "Transparency level (0.0 = fully transparent, 1.0 = fully opaque)",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
    )

    auto_stats = knext.BoolParameter(
        "Auto statistics",
        "Automatically calculate min/max values from non-zero pixels",
        default_value=True,
        is_advanced=True,
    )

    min_value = knext.DoubleParameter(
        "Minimum value",
        "Minimum value for color mapping (ignored if auto statistics is enabled)",
        default_value=0.0,
        is_advanced=True,
    ).rule(knext.OneOf(auto_stats, [False]), knext.Effect.SHOW)

    max_value = knext.DoubleParameter(
        "Maximum value",
        "Maximum value for color mapping (ignored if auto statistics is enabled)",
        default_value=3000,
        is_advanced=True,
    ).rule(knext.OneOf(auto_stats, [False]), knext.Effect.SHOW)

    # Base map options
    base_map = knext.StringParameter(
        "Base map",
        "Base map layer for visualization",
        default_value="OpenStreetMap",
        enum=["OpenStreetMap", "Satellite", "Terrain", "Hybrid"],
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, image_collection_connection
    ):
        import ee
        import geemap.foliumap as geemap
        import logging

        LOGGER = logging.getLogger(__name__)

        # Get image collection and create mosaic (most recent pixel)
        image_collection = image_collection_connection.image_collection
        LOGGER.warning("Processing Image Collection - creating mosaic")
        image = image_collection.mosaic()

        # Get all available band names
        all_band_names = image.bandNames().getInfo()

        # Determine which bands to use
        if self.bands.strip():
            # User specified bands
            requested_bands = [b.strip() for b in self.bands.split(",") if b.strip()]
            # Validate bands exist
            available_requested = [b for b in requested_bands if b in all_band_names]
            missing_bands = [b for b in requested_bands if b not in all_band_names]
            if missing_bands:
                LOGGER.warning(
                    f"Some requested bands not found: {missing_bands}. Available bands: {all_band_names}"
                )
            if not available_requested:
                raise ValueError(
                    f"None of the requested bands found in image. Available bands: {all_band_names}"
                )
            selected_bands = available_requested
        else:
            # Auto-use first available bands
            selected_bands = (
                all_band_names[:3] if len(all_band_names) >= 3 else all_band_names
            )

        # Select only the requested bands
        image = image.select(selected_bands)
        band_names = selected_bands

        LOGGER.warning(f"Visualizing bands: {band_names}")

        # Auto-detect display mode based on number of selected bands
        if len(band_names) >= 3:
            # RGB mode - use first 3 selected bands
            red_band = band_names[0]
            green_band = band_names[1]
            blue_band = band_names[2]

            # Calculate statistics if auto_stats is enabled (for RGB mode)
            if self.auto_stats:
                try:
                    rgb_bands = [red_band, green_band, blue_band]
                    stats = (
                        image.select(rgb_bands)
                        .reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    min_vals = [stats.get(f"{band}_min", 0) for band in rgb_bands]
                    max_vals = [stats.get(f"{band}_max", 0) for band in rgb_bands]

                    min_val = min([v for v in min_vals if v is not None])
                    max_val = max([v for v in max_vals if v is not None])

                    if min_val == 0:
                        non_zero_image = (
                            image.select(rgb_bands)
                            .gt(0)
                            .multiply(image.select(rgb_bands))
                        )
                        non_zero_stats = non_zero_image.reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        ).getInfo()

                        non_zero_min_vals = [
                            non_zero_stats.get(f"{band}_min", min_val)
                            for band in rgb_bands
                        ]
                        min_val = min(
                            [v for v in non_zero_min_vals if v is not None and v > 0]
                            or [min_val]
                        )

                    if max_val <= min_val:
                        max_val = min_val + 1 if min_val is not None else 1

                except Exception as e:
                    min_val = self.min_value
                    max_val = self.max_value
            else:
                min_val = self.min_value
                max_val = self.max_value

            vis = {
                "bands": [red_band, green_band, blue_band],
                "min": min_val,
                "max": max_val,
                "opacity": self.alpha,
            }

        else:
            # Single band mode
            single_band = band_names[0] if band_names else "B1"

            # Parse color palette
            color_list = []
            for color in self.color_palette.split(","):
                color = color.strip()
                if not color.startswith("#"):
                    color = "#" + color
                color_list.append(color)

            # Calculate statistics if auto_stats is enabled
            if self.auto_stats:
                try:
                    stats = (
                        image.select(single_band)
                        .reduceRegion(
                            reducer=ee.Reducer.minMax(),
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    min_val = stats.get(f"{single_band}_min", 0)
                    max_val = stats.get(f"{single_band}_max", 1)

                    if min_val == 0:
                        non_zero_stats = (
                            image.select(single_band)
                            .gt(0)
                            .multiply(image.select(single_band))
                            .reduceRegion(
                                reducer=ee.Reducer.minMax(),
                                geometry=image.geometry(),
                                scale=1000,
                                maxPixels=1e9,
                            )
                            .getInfo()
                        )

                        min_val = non_zero_stats.get(f"{single_band}_min", min_val)
                        if min_val == 0:
                            min_val = stats.get(f"{single_band}_min", 0)

                except Exception as e:
                    min_val = self.min_value
                    max_val = self.max_value
            else:
                min_val = self.min_value
                max_val = self.max_value

            vis = {
                "bands": [single_band],
                "min": min_val,
                "max": max_val,
                "palette": color_list,
                "opacity": self.alpha,
            }

        # Create map
        Map = geemap.Map()

        # Add base map layer
        if self.base_map == "OpenStreetMap":
            Map.add_basemap("OpenStreetMap")
        elif self.base_map == "Satellite":
            Map.add_basemap("Satellite")
        elif self.base_map == "Terrain":
            Map.add_basemap("Terrain")
        elif self.base_map == "Hybrid":
            Map.add_basemap("Hybrid")

        Map.addLayer(image, vis, "GEE Image Collection")

        # center the map automatically based on image bounds
        Map.centerObject(image)
        # replace css and JavaScript paths
        html = Map.get_root().render()

        return knext.view(html)


############################################
# GEE Feature Collection View Node
############################################
@knext.node(
    name="GEE Feature Collection View",
    node_type=knext.NodeType.VISUALIZER,
    icon_path=__NODE_ICON_PATH + "featureView.png",
    category=__category,
    after="",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with embedded feature collection object.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_view(
    name="GEE Feature Collection View",
    description="Showing a map with the GEE feature collection",
    static_resources="libs/leaflet/1.9.3",
)
class ViewNodeGEEFeatureCollection:
    """Visualizes a GEE Feature Collection on a map.

    This node will visualize the given GEE Feature Collection on a map using the [geemap](https://geemap.org/) library.
    This view is highly interactive and allows you to change various aspects of the view within the visualization itself.
    For more information about the supported interactions see the [geemap user guides](https://geemap.org/).

    """

    # Feature collection visualization parameters
    color_column = knext.StringParameter(
        "Color column",
        "Column name to use for coloring features (leave empty for uniform color)",
        default_value="",
    )

    color_palette = knext.StringParameter(
        "Fill color palette",
        """Comma-separated color codes for feature coloring. Supports hex codes with or without # prefix.
        
        Common color combinations:

        • Default: FF0000,00FF00,0000FF (red-green-blue)

        • Terrain: 8B4513,FFFF00,00FF00 (brown-yellow-green)

        • Political: FF6B6B,4ECDC4,45B7D1,96CEB4,FFEAA7 (red-teal-blue-green-yellow)
        
        • Simple: FF0000,0000FF (red-blue)

        Examples: 'FF0000,00FF00,0000FF' or '#FF0000,#00FF00,#0000FF'""",
        default_value="FF0000,00FF00,0000FF",
    )

    fill_opacity = knext.DoubleParameter(
        "Fill opacity",
        "Opacity of the fill color (0.0 = transparent, 1.0 = opaque)",
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
    )

    stroke_color = knext.StringParameter(
        "Stroke color",
        "Hex color code for polygon/line stroke (e.g., '000000' or '#000000')",
        default_value="000000",
    )

    stroke_width = knext.IntParameter(
        "Stroke width",
        "Width of the stroke line in pixels",
        default_value=2,
        min_value=1,
        max_value=10,
    )

    stroke_opacity = knext.DoubleParameter(
        "Stroke opacity",
        "Opacity of the stroke color (0.0 = transparent, 1.0 = opaque)",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
    )

    point_radius = knext.IntParameter(
        "Point radius",
        "Radius of point features in pixels",
        default_value=5,
        min_value=1,
        max_value=20,
    )

    num_classes = knext.IntParameter(
        "Number of classes",
        "Number of color classes for classification (auto-adjusts for discrete values)",
        default_value=5,
        min_value=3,
        is_advanced=True,
    )

    # Base map options
    base_map = knext.StringParameter(
        "Base map",
        "Base map layer for visualization",
        default_value="OpenStreetMap",
        enum=["OpenStreetMap", "Satellite", "Terrain", "Hybrid"],
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, fc_connection):
        import ee
        import geemap.foliumap as geemap
        import logging

        LOGGER = logging.getLogger(__name__)

        feature_collection = fc_connection.feature_collection

        color_list = []
        for color in self.color_palette.split(","):
            color = color.strip()
            # Add # prefix if not present
            if not color.startswith("#"):
                color = "#" + color
            color_list.append(color)

        vis_params = {
            "color": (
                self.stroke_color
                if self.stroke_color.startswith("#")
                else f"#{self.stroke_color}"
            ),
        }

        palette = [
            c.strip().lstrip("#") for c in self.color_palette.split(",") if c.strip()
        ]

        # ---------- build a color ramp from user palette ----------
        def _hex_to_rgb(h):  # 'RRGGBB' -> (r,g,b)
            h = h.strip().lstrip("#")
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

        def _rgb_to_hex(rgb):  # (r,g,b) -> 'RRGGBB'
            return "".join(f"{max(0,min(255,v)):02X}" for v in rgb)

        def make_ramp_hex(control_colors, n_steps):
            """Linear RGB interpolation to get n_steps colors; control_colors as ['FF0000','00FF00',...]"""
            cs = [c.strip().lstrip("#") for c in control_colors if c.strip()]
            if not cs:  # fallback
                return ["FF0000"] * max(1, n_steps)
            if len(cs) == 1:
                return [cs[0]] * max(1, n_steps)

            segs = len(cs) - 1
            out = []
            for i in range(n_steps):
                t = 0 if n_steps == 1 else i / (n_steps - 1)  # 0..1
                s = min(int(t * segs), segs - 1)  # segment index
                t0, t1 = s / segs, (s + 1) / segs
                lt = 0 if t1 == t0 else (t - t0) / (t1 - t0)  # local t in [0,1]
                c0, c1 = _hex_to_rgb(cs[s]), _hex_to_rgb(cs[s + 1])
                rgb = tuple(int(round(c0[k] * (1 - lt) + c1[k] * lt)) for k in range(3))
                out.append(_rgb_to_hex(rgb))
            return out

        # User-controllable steps: number of colors for classification/categories
        N_CLASSES = max(
            5, len(palette)
        )  # e.g. at least 5 levels; can be fixed to 7, 9, etc.
        ramp = make_ramp_hex(
            palette, N_CLASSES
        )  # e.g. ['FF0000', 'BF3F00', ..., '0000FF']
        ramp_ee = ee.List(ramp)

        def with_alpha(hexrgb, alpha01):
            a = max(0, min(1, float(alpha01)))
            return hexrgb + f"{int(round(a*255)):02X}"

        # GEE version of with_alpha for server-side processing
        def with_alpha_ee(hexrgb, alpha01):
            a = ee.Number(alpha01).max(0).min(1)
            alpha_hex = a.multiply(255).round().format("%02X")
            return ee.String(hexrgb).cat(alpha_hex)

        # Styling function for equal interval classification

        def style_by_equal_interval(
            fc, column, ramp_ee, fill_opacity, base_style, num_classes
        ):
            # Dynamic equal interval classification
            mm = fc.reduceColumns(ee.Reducer.minMax(), [column])  # {'min':..,'max':..}
            vmin, vmax = ee.Number(mm.get("min")), ee.Number(mm.get("max"))
            step = vmax.subtract(vmin).divide(ee.Number(num_classes))

            def class_index(v):
                # floor((v - vmin)/step) truncated to [0, k-1]
                return (
                    ee.Number(v.subtract(vmin).divide(step).floor())
                    .max(0)
                    .min(ee.Number(num_classes).subtract(1))
                )

            def set_style(f):
                v = ee.Number(f.get(column))
                idx = class_index(v)
                col = ee.String(ramp_ee.get(idx))
                sty = ee.Dictionary(
                    {
                        "fillColor": with_alpha_ee(
                            col, fill_opacity
                        ),  # Key: add alpha to fill
                        "color": with_alpha_ee(
                            stroke, ee.Number(self.stroke_opacity)
                        ),  # Optional: add alpha to stroke
                        "width": self.stroke_width,
                        "pointRadius": self.point_radius,
                    }
                )
                return f.set("style", sty)

            return fc.filter(ee.Filter.notNull([column])).map(set_style).style(
                styleProperty="style"
            ), ee.List.sequence(1, ee.Number(num_classes).subtract(1)).map(
                lambda i: vmin.add(step.multiply(i))
            )  # Return break list for legend

        stroke = self.stroke_color.lstrip("#")
        base_style = {
            "color": stroke,
            "width": self.stroke_width,
        }

        # Create map
        Map = geemap.Map()

        # Add base map layer
        if self.base_map == "OpenStreetMap":
            Map.add_basemap("OpenStreetMap")
        elif self.base_map == "Satellite":
            Map.add_basemap("Satellite")
        elif self.base_map == "Terrain":
            Map.add_basemap("Terrain")
        elif self.base_map == "Hybrid":
            Map.add_basemap("Hybrid")

        # Apply gradient coloring system
        layer_to_add = None

        if self.color_column:
            # Auto-detect data type
            try:
                # Get column data type sample
                sample_value = feature_collection.first().get(self.color_column)
                sample_info = sample_value.getInfo()

                # Check if numeric type
                is_numeric = isinstance(sample_info, (int, float)) and not isinstance(
                    sample_info, bool
                )

                if is_numeric:
                    # Numeric type: use equal interval classification
                    styled_img, breaks = style_by_equal_interval(
                        feature_collection,
                        self.color_column,
                        ramp_ee,
                        self.fill_opacity,
                        base_style,
                        self.num_classes,
                    )
                    Map.addLayer(
                        styled_img, {}, f"{self.color_column} (equal interval)"
                    )

                else:
                    # Non-numeric type (string): use default fill color
                    fill = with_alpha(
                        palette[0] if palette else "FF0000", self.fill_opacity
                    )
                    layer_to_add = feature_collection.style(
                        **dict(base_style, fillColor=fill)
                    )
                    Map.addLayer(layer_to_add, {}, "Feature Collection")

                layer_to_add = None  # Already added through Map.addLayer

            except Exception as e:
                # If column not found or other error: use default fill color
                fill = with_alpha(
                    palette[0] if palette else "FF0000", self.fill_opacity
                )
                layer_to_add = feature_collection.style(
                    **dict(base_style, fillColor=fill)
                )
                Map.addLayer(layer_to_add, {}, "Feature Collection")
        else:
            # Uniform coloring
            fill = with_alpha(palette[0] if palette else "FF0000", self.fill_opacity)
            layer_to_add = feature_collection.style(**dict(base_style, fillColor=fill))
            Map.addLayer(layer_to_add, {}, "Feature Collection")

        # Auto center the map to fit all features
        Map.centerObject(feature_collection)

        # Get map HTML
        html = Map.get_root().render()

        return knext.view(html)
