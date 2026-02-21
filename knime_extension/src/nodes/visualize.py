import re

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
    description="Display GEE images, image collections, and feature collections on a map.",
    icon="icons/visualize.png",
    after="focal",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/visualize/"

# CSS named colors (W3C / CSS Color Module). Full list:
# https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
# https://www.w3.org/TR/css-color-4/#named-colors
_CSS_NAMED_COLORS = {
    "aliceblue": "F0F8FF",
    "antiquewhite": "FAEBD7",
    "aqua": "00FFFF",
    "aquamarine": "7FFFD4",
    "azure": "F0FFFF",
    "beige": "F5F5DC",
    "bisque": "FFE4C4",
    "black": "000000",
    "blanchedalmond": "FFEBCD",
    "blue": "0000FF",
    "blueviolet": "8A2BE2",
    "brown": "A52A2A",
    "burlywood": "DEB887",
    "cadetblue": "5F9EA0",
    "chartreuse": "7FFF00",
    "chocolate": "D2691E",
    "coral": "FF7F50",
    "cornflowerblue": "6495ED",
    "cornsilk": "FFF8DC",
    "crimson": "DC143C",
    "cyan": "00FFFF",
    "darkblue": "00008B",
    "darkcyan": "008B8B",
    "darkgoldenrod": "B8860B",
    "darkgray": "A9A9A9",
    "darkgreen": "006400",
    "darkgrey": "A9A9A9",
    "darkkhaki": "BDB76B",
    "darkmagenta": "8B008B",
    "darkolivegreen": "556B2F",
    "darkorange": "FF8C00",
    "darkorchid": "9932CC",
    "darkred": "8B0000",
    "darksalmon": "E9967A",
    "darkseagreen": "8FBC8F",
    "darkslateblue": "483D8B",
    "darkslategray": "2F4F4F",
    "darkslategrey": "2F4F4F",
    "darkturquoise": "00CED1",
    "darkviolet": "9400D3",
    "deeppink": "FF1493",
    "deepskyblue": "00BFFF",
    "dimgray": "696969",
    "dimgrey": "696969",
    "dodgerblue": "1E90FF",
    "firebrick": "B22222",
    "floralwhite": "FFFAF0",
    "forestgreen": "228B22",
    "fuchsia": "FF00FF",
    "gainsboro": "DCDCDC",
    "ghostwhite": "F8F8FF",
    "gold": "FFD700",
    "goldenrod": "DAA520",
    "gray": "808080",
    "green": "008000",
    "greenyellow": "ADFF2F",
    "grey": "808080",
    "honeydew": "F0FFF0",
    "hotpink": "FF69B4",
    "indianred": "CD5C5C",
    "indigo": "4B0082",
    "ivory": "FFFFF0",
    "khaki": "F0E68C",
    "lavender": "E6E6FA",
    "lavenderblush": "FFF0F5",
    "lawngreen": "7CFC00",
    "lemonchiffon": "FFFACD",
    "lightblue": "ADD8E6",
    "lightcoral": "F08080",
    "lightcyan": "E0FFFF",
    "lightgoldenrodyellow": "FAFAD2",
    "lightgray": "D3D3D3",
    "lightgreen": "90EE90",
    "lightgrey": "D3D3D3",
    "lightpink": "FFB6C1",
    "lightsalmon": "FFA07A",
    "lightseagreen": "20B2AA",
    "lightskyblue": "87CEFA",
    "lightslategray": "778899",
    "lightslategrey": "778899",
    "lightsteelblue": "B0C4DE",
    "lightyellow": "FFFFE0",
    "lime": "00FF00",
    "limegreen": "32CD32",
    "linen": "FAF0E6",
    "magenta": "FF00FF",
    "maroon": "800000",
    "mediumaquamarine": "66CDAA",
    "mediumblue": "0000CD",
    "mediumorchid": "BA55D3",
    "mediumpurple": "9370DB",
    "mediumseagreen": "3CB371",
    "mediumslateblue": "7B68EE",
    "mediumspringgreen": "00FA9A",
    "mediumturquoise": "48D1CC",
    "mediumvioletred": "C71585",
    "midnightblue": "191970",
    "mintcream": "F5FFFA",
    "mistyrose": "FFE4E1",
    "moccasin": "FFE4B5",
    "navajowhite": "FFDEAD",
    "navy": "000080",
    "oldlace": "FDF5E6",
    "olive": "808000",
    "olivedrab": "6B8E23",
    "orange": "FFA500",
    "orangered": "FF4500",
    "orchid": "DA70D6",
    "palegoldenrod": "EEE8AA",
    "palegreen": "98FB98",
    "paleturquoise": "AFEEEE",
    "palevioletred": "DB7093",
    "papayawhip": "FFEFD5",
    "peachpuff": "FFDAB9",
    "peru": "CD853F",
    "pink": "FFC0CB",
    "plum": "DDA0DD",
    "powderblue": "B0E0E6",
    "purple": "800080",
    "red": "FF0000",
    "rosybrown": "BC8F8F",
    "royalblue": "4169E1",
    "saddlebrown": "8B4513",
    "salmon": "FA8072",
    "sandybrown": "F4A460",
    "seagreen": "2E8B57",
    "seashell": "FFF5EE",
    "sienna": "A0522D",
    "silver": "C0C0C0",
    "skyblue": "87CEEB",
    "slateblue": "6A5ACD",
    "slategray": "708090",
    "slategrey": "708090",
    "snow": "FFFAFA",
    "springgreen": "00FF7F",
    "steelblue": "4682B4",
    "tan": "D2B48C",
    "teal": "008080",
    "thistle": "D8BFD8",
    "tomato": "FF6347",
    "turquoise": "40E0D0",
    "violet": "EE82EE",
    "wheat": "F5DEB3",
    "white": "FFFFFF",
    "whitesmoke": "F5F5F5",
    "yellow": "FFFF00",
    "yellowgreen": "9ACD32",
}


def _normalize_color_to_hex(token):
    """Convert a color token to hex with #. Accepts 6-digit hex (with or without #) or CSS named color."""
    if not token or not isinstance(token, str):
        raise ValueError(f"Invalid color: {token!r}")
    s = token.strip()
    if not s:
        raise ValueError("Empty color token")
    # Already hex: 6 hex digits, optional leading #
    hex_only = s.lstrip("#")
    if re.match(r"^[0-9A-Fa-f]{6}$", hex_only):
        return "#" + hex_only.upper()
    # CSS named color (case-insensitive)
    key = s.lower()
    if key in _CSS_NAMED_COLORS:
        return "#" + _CSS_NAMED_COLORS[key]
    raise ValueError(
        f"Unknown color: {s!r}. Use 6-digit hex (e.g. FF0000) or a CSS color name (e.g. red). "
        "See: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color"
    )


############################################
# Helper: center map on EE object with auto zoom (skip for global extent)
############################################


def _center_map_on_object(Map, ee_object):
    """Center map on EE object (image/geometry). Use fit_bounds from geometry.
    If extent is global (covers whole world), use default centerObject behavior."""
    import ee

    try:
        geom = ee_object.geometry() if hasattr(ee_object, "geometry") else ee_object
        bounds_info = geom.bounds().getInfo()
        if not bounds_info or "coordinates" not in bounds_info:
            Map.centerObject(ee_object)
            return
        coords = bounds_info.get("coordinates", [[]])[0]
        if not coords:
            Map.centerObject(ee_object)
            return
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        lon_span = abs(max_lon - min_lon)
        lat_span = abs(max_lat - min_lat)
        if lon_span > 350 or lat_span > 170:
            Map.centerObject(ee_object)
            return
        # folium fit_bounds: [[south, west], [north, east]] = [[min_lat, min_lon], [max_lat, max_lon]]
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        Map.fit_bounds(bounds)
    except Exception:
        Map.centerObject(ee_object)


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

    **Band Selection:**
    - Specify which bands to visualize (e.g., 'B4,B3,B2' for RGB)
    - Leave empty to auto-use first 3 bands for RGB or first band for single-band
    - Common RGB combinations:
      - Landsat: 'B4,B3,B2' (True Color)
      - Sentinel-2: 'B4,B3,B2' (True Color)
      - False Color: 'B8,B4,B3' (Sentinel-2 vegetation)

    This node provides two visualization modes for GEE Images:

    1. **Single Band Mode**: For 1-2 selected bands, displays the first band with color mapping
    2. **RGB Mode**: For 3+ selected bands, uses first 3 bands as RGB channels

    **Visualization Features**:
    - **Statistics Modes**:
      - **Auto Min/Max**: Automatically calculates min/max for each band independently
      - **Auto Quartiles**: Automatically calculates Q1/Q3 for each band independently (robust to outliers)
      - **Manual**: Manually specify min/max values (single value or comma-separated list per band)
    - **Per-Band Visualization**: In RGB mode, each band can have independent min/max values
    - **Color Palettes**: Comma-separated colors as hex or CSS color names (e.g. red, blue). See: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    - **Transparency**: Adjustable alpha channel for overlay visualization
    - **Base Maps**: Choose from OpenStreetMap, Satellite, Terrain, or Hybrid backgrounds

    **Usage**:
    - Specify bands to visualize (e.g., 'B4,B3,B2' for RGB visualization)
    - Leave bands empty to auto-use first available bands
    - Choose statistics mode: Auto Min/Max (default), Auto Quartiles (robust), or Manual
    - In Manual mode, provide min/max as single value (all bands) or comma-separated list (per-band)
    - Example for PCA: bands='pc1,pc3,pc4', mode='manual', min='-455.09,-2.206,-4.53', max='-417.59,-1.3,-4.18'

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
        """Comma-separated color codes for gradient. Supports **hex** (6 digits, with or without #) and **CSS color names** (e.g. red, blue, darkgreen). Color names follow the CSS named color dictionary: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Common gradient combinations (hex or names):
        • Terrain: 000080,00FFFF,00FF00,FFFF00,FF0000 or navy,aqua,lime,yellow,red
        • Heatmap: 0000FF,00FFFF,00FF00,FFFF00,FF0000 (blue-cyan-green-yellow-red)
        • Vegetation: 8B4513,FFFF00,00FF00 (brown-yellow-green)
        • Simple: 000000,FFFFFF or black,white

        Examples: '000080,00FF00,FF0000' or 'navy,green,red'""",
        default_value="000000,FFFFFF",
    )

    alpha = knext.DoubleParameter(
        "Alpha (Transparency)",
        "Transparency level (0.0 = fully transparent, 1.0 = fully opaque)",
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
    )
    # Statistics mode
    stats_mode = knext.StringParameter(
        "Statistics mode",
        """How to determine visualization ranges:
        - 'autoMinMax': Automatically calculate min/max for each band independently
        - 'autoQuartiles': Automatically calculate Q1/Q3 for each band independently (robust to outliers)
        - 'manual': Manually specify min/max values (single value or comma-separated list per band)""",
        default_value="autoMinMax",
        enum=["autoMinMax", "autoQuartiles", "manual"],
        is_advanced=True,
    )

    min_value = knext.StringParameter(
        "Minimum value(s)",
        """Minimum value(s) for color mapping. Only used when Statistics mode is 'manual'.
        
        Can be a single value (applied to all bands) or comma-separated list (one value per band).
        
        Examples:
        - Single value: '0.0' (all bands use same min)
        - Per-band values: '-455.09,-2.206,-4.53' (for 3 bands: pc1, pc3, pc4)
        
        For RGB visualization, provide 1 value (all bands) or 3 values (one per band).""",
        default_value="0.0",
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

    max_value = knext.StringParameter(
        "Maximum value(s)",
        """Maximum value(s) for color mapping. Only used when Statistics mode is 'manual'.
        
        Can be a single value (applied to all bands) or comma-separated list (one value per band).
        
        Examples:
        - Single value: '0.3' (all bands use same max)
        - Per-band values: '-417.59,-1.3,-4.18' (for 3 bands: pc1, pc3, pc4)
        
        For RGB visualization, provide 1 value (all bands) or 3 values (one per band).""",
        default_value="0.3",
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

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

        # Get all available band names
        all_band_names = image.bandNames().getInfo()

        # Determine which bands to use
        if self.bands and self.bands.strip():
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
            # Auto-use first available bands (default behavior)
            selected_bands = (
                all_band_names[:3] if len(all_band_names) >= 3 else all_band_names
            )

        # Select only the requested bands
        image = image.select(selected_bands)
        band_names = selected_bands

        # Auto-detect display mode based on number of selected bands
        if len(band_names) >= 3:
            # RGB mode - use first 3 selected bands
            red_band = band_names[0]
            green_band = band_names[1]
            blue_band = band_names[2]

            # Calculate statistics based on selected mode (for RGB mode)
            rgb_bands = [red_band, green_band, blue_band]

            if self.stats_mode == "autoMinMax":
                try:
                    # Calculate min/max for each band independently
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

                    # Get min and max for each band independently
                    min_vals = [stats.get(f"{band}_min", 0) for band in rgb_bands]
                    max_vals = [stats.get(f"{band}_max", 0) for band in rgb_bands]

                    # Filter out zero values for better visualization
                    if any(v == 0 for v in min_vals):
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
                            non_zero_stats.get(f"{band}_min", min_vals[i])
                            for i, band in enumerate(rgb_bands)
                        ]
                        min_vals = [
                            v if v > 0 else min_vals[i]
                            for i, v in enumerate(non_zero_min_vals)
                        ]

                    # Ensure max > min for each band
                    for i in range(len(rgb_bands)):
                        if max_vals[i] <= min_vals[i]:
                            max_vals[i] = (
                                min_vals[i] + 1 if min_vals[i] is not None else 1
                            )

                    min_val = min_vals  # List of 3 values
                    max_val = max_vals  # List of 3 values

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto min/max: {e}, using defaults"
                    )
                    min_val = [0.0, 0.0, 0.0]
                    max_val = [0.3, 0.3, 0.3]

            elif self.stats_mode == "autoQuartiles":
                try:
                    # Calculate Q1 and Q3 for each band independently
                    stats = (
                        image.select(rgb_bands)
                        .reduceRegion(
                            reducer=ee.Reducer.percentile([25, 75]),  # Q1 and Q3
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    # Get Q1 (p25) and Q3 (p75) for each band independently
                    min_vals = [stats.get(f"{band}_p25", 0) for band in rgb_bands]  # Q1
                    max_vals = [stats.get(f"{band}_p75", 0) for band in rgb_bands]  # Q3

                    # Ensure max > min for each band
                    for i in range(len(rgb_bands)):
                        if max_vals[i] <= min_vals[i]:
                            max_vals[i] = (
                                min_vals[i] + 1 if min_vals[i] is not None else 1
                            )

                    min_val = min_vals  # List of 3 values
                    max_val = max_vals  # List of 3 values

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto quartiles: {e}, using defaults"
                    )
                    min_val = [0.0, 0.0, 0.0]
                    max_val = [0.3, 0.3, 0.3]

            else:  # manual mode
                # Parse manual values
                min_str = str(self.min_value).strip()
                max_str = str(self.max_value).strip()

                min_list = [float(v.strip()) for v in min_str.split(",") if v.strip()]
                max_list = [float(v.strip()) for v in max_str.split(",") if v.strip()]

                if len(min_list) == 1:
                    # Single value - apply to all bands
                    min_val = [min_list[0]] * 3
                elif len(min_list) == 3:
                    # Per-band values
                    min_val = min_list
                else:
                    raise ValueError(
                        f"Min values must be 1 value (for all bands) or 3 values (one per band). "
                        f"Got {len(min_list)} values: {min_list}"
                    )

                if len(max_list) == 1:
                    # Single value - apply to all bands
                    max_val = [max_list[0]] * 3
                elif len(max_list) == 3:
                    # Per-band values
                    max_val = max_list
                else:
                    raise ValueError(
                        f"Max values must be 1 value (for all bands) or 3 values (one per band). "
                        f"Got {len(max_list)} values: {max_list}"
                    )

            vis = {
                "bands": [red_band, green_band, blue_band],
                "min": min_val,  # Can be single value or list of 3 values
                "max": max_val,  # Can be single value or list of 3 values
                "opacity": self.alpha,
            }
            # LOGGER.warning(f"RGB mode: {red_band}, {green_band}, {blue_band}, range: {min_val}-{max_val}")

        else:
            # Single band mode - use first selected band
            single_band = band_names[0] if band_names else "B1"

            # Parse color palette (hex or CSS named colors)
            color_list = []
            for color in self.color_palette.split(","):
                color_list.append(_normalize_color_to_hex(color))

            # Calculate statistics based on selected mode
            if self.stats_mode == "autoMinMax":
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

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto min/max: {e}, using defaults"
                    )
                    min_val = 0.0
                    max_val = 0.3

            elif self.stats_mode == "autoQuartiles":
                try:
                    # Calculate Q1 and Q3 for the single band
                    stats = (
                        image.select(single_band)
                        .reduceRegion(
                            reducer=ee.Reducer.percentile([25, 75]),  # Q1 and Q3
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    min_val = stats.get(f"{single_band}_p25", 0)  # Q1
                    max_val = stats.get(f"{single_band}_p75", 1)  # Q3

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto quartiles: {e}, using defaults"
                    )
                    min_val = 0.0
                    max_val = 0.3

            else:  # manual mode
                # Parse manual values (use first value if multiple provided)
                min_str = str(self.min_value).strip()
                max_str = str(self.max_value).strip()
                min_list = [float(v.strip()) for v in min_str.split(",") if v.strip()]
                max_list = [float(v.strip()) for v in max_str.split(",") if v.strip()]
                min_val = min_list[0] if min_list else 0.0
                max_val = max_list[0] if max_list else 0.3

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

        _center_map_on_object(Map, image)
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
    - **Statistics Modes**:
      - **Auto Min/Max**: Automatically calculates min/max for each band independently
      - **Auto Quartiles**: Automatically calculates Q1/Q3 for each band independently (robust to outliers)
      - **Manual**: Manually specify min/max values (single value or comma-separated list per band)
    - **Per-Band Visualization**: In RGB mode, each band can have independent min/max values
    - **Color Palettes**: Comma-separated colors as hex or CSS color names (e.g. red, blue). See: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
    - **Transparency**: Adjustable alpha channel for overlay visualization
    - **Base Maps**: Choose from OpenStreetMap, Satellite, Terrain, or Hybrid backgrounds

    **Usage**:
    - Specify bands to visualize (e.g., 'B4,B3,B2' for RGB visualization)
    - Leave bands empty to auto-use first available bands
    - Choose statistics mode: Auto Min/Max (default), Auto Quartiles (robust), or Manual
    - In Manual mode, provide min/max as single value (all bands) or comma-separated list (per-band)
    - Example for PCA: bands='pc1,pc3,pc4', mode='manual', min='-455.09,-2.206,-4.53', max='-417.59,-1.3,-4.18'

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

    # Statistics mode
    stats_mode = knext.StringParameter(
        "Statistics mode",
        """How to determine visualization ranges:
        - 'autoMinMax': Automatically calculate min/max for each band independently
        - 'autoQuartiles': Automatically calculate Q1/Q3 for each band independently (robust to outliers)
        - 'manual': Manually specify min/max values (single value or comma-separated list per band)""",
        default_value="autoMinMax",
        enum=["autoMinMax", "autoQuartiles", "manual"],
        is_advanced=True,
    )

    min_value = knext.StringParameter(
        "Minimum value(s)",
        """Minimum value(s) for color mapping. Only used when Statistics mode is 'manual'.
        
        Can be a single value (applied to all bands) or comma-separated list (one value per band).
        
        Examples:
        - Single value: '0.0' (all bands use same min)
        - Per-band values: '-455.09,-2.206,-4.53' (for 3 bands: pc1, pc3, pc4)
        
        For RGB visualization, provide 1 value (all bands) or 3 values (one per band).""",
        default_value="0.0",
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

    max_value = knext.StringParameter(
        "Maximum value(s)",
        """Maximum value(s) for color mapping. Only used when Statistics mode is 'manual'.
        
        Can be a single value (applied to all bands) or comma-separated list (one value per band).
        
        Examples:
        - Single value: '3000' (all bands use same max)
        - Per-band values: '-417.59,-1.3,-4.18' (for 3 bands: pc1, pc3, pc4)
        
        For RGB visualization, provide 1 value (all bands) or 3 values (one per band).""",
        default_value="3000",
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

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

            # Calculate statistics based on selected mode (for RGB mode)
            rgb_bands = [red_band, green_band, blue_band]

            if self.stats_mode == "autoMinMax":
                try:
                    # Calculate min/max for each band independently
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

                    # Get min and max for each band independently
                    min_vals = [stats.get(f"{band}_min", 0) for band in rgb_bands]
                    max_vals = [stats.get(f"{band}_max", 0) for band in rgb_bands]

                    # Filter out zero values for better visualization
                    if any(v == 0 for v in min_vals):
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
                            non_zero_stats.get(f"{band}_min", min_vals[i])
                            for i, band in enumerate(rgb_bands)
                        ]
                        min_vals = [
                            v if v > 0 else min_vals[i]
                            for i, v in enumerate(non_zero_min_vals)
                        ]

                    # Ensure max > min for each band
                    for i in range(len(rgb_bands)):
                        if max_vals[i] <= min_vals[i]:
                            max_vals[i] = (
                                min_vals[i] + 1 if min_vals[i] is not None else 1
                            )

                    min_val = min_vals  # List of 3 values
                    max_val = max_vals  # List of 3 values

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto min/max: {e}, using defaults"
                    )
                    min_val = [0.0, 0.0, 0.0]
                    max_val = [3000.0, 3000.0, 3000.0]

            elif self.stats_mode == "autoQuartiles":
                try:
                    # Calculate Q1 and Q3 for each band independently
                    stats = (
                        image.select(rgb_bands)
                        .reduceRegion(
                            reducer=ee.Reducer.percentile([25, 75]),  # Q1 and Q3
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    # Get Q1 (p25) and Q3 (p75) for each band independently
                    min_vals = [stats.get(f"{band}_p25", 0) for band in rgb_bands]  # Q1
                    max_vals = [stats.get(f"{band}_p75", 0) for band in rgb_bands]  # Q3

                    # Ensure max > min for each band
                    for i in range(len(rgb_bands)):
                        if max_vals[i] <= min_vals[i]:
                            max_vals[i] = (
                                min_vals[i] + 1 if min_vals[i] is not None else 1
                            )

                    min_val = min_vals  # List of 3 values
                    max_val = max_vals  # List of 3 values

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto quartiles: {e}, using defaults"
                    )
                    min_val = [0.0, 0.0, 0.0]
                    max_val = [3000.0, 3000.0, 3000.0]

            else:  # manual mode
                # Parse manual values
                min_str = str(self.min_value).strip()
                max_str = str(self.max_value).strip()

                min_list = [float(v.strip()) for v in min_str.split(",") if v.strip()]
                max_list = [float(v.strip()) for v in max_str.split(",") if v.strip()]

                if len(min_list) == 1:
                    # Single value - apply to all bands
                    min_val = [min_list[0]] * 3
                elif len(min_list) == 3:
                    # Per-band values
                    min_val = min_list
                else:
                    raise ValueError(
                        f"Min values must be 1 value (for all bands) or 3 values (one per band). "
                        f"Got {len(min_list)} values: {min_list}"
                    )

                if len(max_list) == 1:
                    # Single value - apply to all bands
                    max_val = [max_list[0]] * 3
                elif len(max_list) == 3:
                    # Per-band values
                    max_val = max_list
                else:
                    raise ValueError(
                        f"Max values must be 1 value (for all bands) or 3 values (one per band). "
                        f"Got {len(max_list)} values: {max_list}"
                    )

            vis = {
                "bands": [red_band, green_band, blue_band],
                "min": min_val,  # Can be single value or list of 3 values
                "max": max_val,  # Can be single value or list of 3 values
                "opacity": self.alpha,
            }

        else:
            # Single band mode
            single_band = band_names[0] if band_names else "B1"

            # Parse color palette (hex or CSS named colors)
            color_list = []
            for color in self.color_palette.split(","):
                color_list.append(_normalize_color_to_hex(color))

            # Calculate statistics based on selected mode
            if self.stats_mode == "autoMinMax":
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
                    LOGGER.warning(
                        f"Failed to calculate auto min/max: {e}, using defaults"
                    )
                    min_val = 0.0
                    max_val = 3000.0

            elif self.stats_mode == "autoQuartiles":
                try:
                    # Calculate Q1 and Q3 for the single band
                    stats = (
                        image.select(single_band)
                        .reduceRegion(
                            reducer=ee.Reducer.percentile([25, 75]),  # Q1 and Q3
                            geometry=image.geometry(),
                            scale=1000,
                            maxPixels=1e9,
                        )
                        .getInfo()
                    )

                    min_val = stats.get(f"{single_band}_p25", 0)  # Q1
                    max_val = stats.get(f"{single_band}_p75", 1)  # Q3

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to calculate auto quartiles: {e}, using defaults"
                    )
                    min_val = 0.0
                    max_val = 3000.0

            else:  # manual mode
                # Parse manual values (use first value if multiple provided)
                min_str = str(self.min_value).strip()
                max_str = str(self.max_value).strip()
                min_list = [float(v.strip()) for v in min_str.split(",") if v.strip()]
                max_list = [float(v.strip()) for v in max_str.split(",") if v.strip()]
                min_val = min_list[0] if min_list else 0.0
                max_val = max_list[0] if max_list else 3000.0

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
        """Comma-separated color codes for feature coloring. Supports **hex** (6 digits, with or without #) and **CSS color names** (e.g. red, blue, darkgreen). Full list: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color

        Examples: 'FF0000,00FF00,0000FF' or 'red,green,blue'""",
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
        "Polygon/line stroke color: hex (e.g. 000000 or #000000) or CSS color name (e.g. black). See: https://developer.mozilla.org/en-US/docs/Web/CSS/named-color",
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

    stats_mode = knext.StringParameter(
        "Statistics mode",
        "How to determine min/max for the color scale (numeric color column only): "
        "autoMinMax (from data), autoQuartiles (Q1–Q3, robust to outliers), or manual (you set min/max).",
        default_value="autoMinMax",
        enum=["autoMinMax", "autoQuartiles", "manual"],
        is_advanced=True,
    )

    min_value = knext.DoubleParameter(
        "Minimum value",
        "Minimum value for color scale. Only used when Statistics mode is 'manual'.",
        default_value=0.0,
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

    max_value = knext.DoubleParameter(
        "Maximum value",
        "Maximum value for color scale. Only used when Statistics mode is 'manual'.",
        default_value=100.0,
        is_advanced=True,
    ).rule(knext.OneOf(stats_mode, ["manual"]), knext.Effect.SHOW)

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

        # Parse fill palette (hex or CSS named colors)
        color_list = []
        for color in self.color_palette.split(","):
            color_list.append(_normalize_color_to_hex(color))

        stroke_hex = _normalize_color_to_hex(self.stroke_color)
        vis_params = {"color": stroke_hex}

        # For ramp interpolation we need hex without # (RRGGBB)
        palette = [c.lstrip("#") for c in color_list]

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
            fc,
            column,
            ramp_ee,
            fill_opacity,
            base_style,
            num_classes,
            vmin_ee=None,
            vmax_ee=None,
        ):
            # Use provided min/max or compute from FC
            if vmin_ee is not None and vmax_ee is not None:
                vmin = ee.Number(vmin_ee)
                vmax = ee.Number(vmax_ee)
            else:
                mm = fc.reduceColumns(ee.Reducer.minMax(), [column])
                vmin = ee.Number(mm.get("min"))
                vmax = ee.Number(mm.get("max"))
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

        stroke = stroke_hex.lstrip("#")
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
                    # Numeric type: use equal interval classification; get vmin/vmax by stats_mode
                    vmin_ee, vmax_ee = None, None
                    if self.stats_mode == "manual":
                        vmin_ee = ee.Number(self.min_value)
                        vmax_ee = ee.Number(self.max_value)
                    elif self.stats_mode == "autoMinMax":
                        try:
                            mm = feature_collection.reduceColumns(
                                ee.Reducer.minMax(), [self.color_column]
                            ).getInfo()
                            vmin_ee = ee.Number(mm.get("min", 0))
                            vmax_ee = ee.Number(mm.get("max", 1))
                        except Exception as e:
                            LOGGER.warning(
                                "Failed to get min/max for %s: %s; using 0 and 1",
                                self.color_column,
                                e,
                            )
                            vmin_ee = ee.Number(0)
                            vmax_ee = ee.Number(1)
                    elif self.stats_mode == "autoQuartiles":
                        try:
                            pp = feature_collection.reduceColumns(
                                ee.Reducer.percentile([25, 75]), [self.color_column]
                            ).getInfo()
                            vmin_ee = ee.Number(pp.get("p25", 0))
                            vmax_ee = ee.Number(pp.get("p75", 1))
                        except Exception as e:
                            LOGGER.warning(
                                "Failed to get quartiles for %s: %s; using 0 and 1",
                                self.color_column,
                                e,
                            )
                            vmin_ee = ee.Number(0)
                            vmax_ee = ee.Number(1)
                    styled_img, breaks = style_by_equal_interval(
                        feature_collection,
                        self.color_column,
                        ramp_ee,
                        self.fill_opacity,
                        base_style,
                        self.num_classes,
                        vmin_ee=vmin_ee,
                        vmax_ee=vmax_ee,
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
