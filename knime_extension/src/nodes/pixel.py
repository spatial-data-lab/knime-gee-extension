"""
GEE Pixel Transform nodes for KNIME.
Pixel-level image operations: band math, indices, aggregation, normalization, PCA, unmixing, color transforms.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
    gee_image_collection_port_type,
    gee_feature_collection_port_type,
    gee_array_port_type,
    GEEArrayConnectionObject,
)

# Category for GEE Advanced Pixel Transformation nodes
__category = knext.category(
    path="/community/gee",
    level_id="pixeltransform",
    name="GEE Pixel Transform",
    description="Pixel-level image operations: band math, indices, aggregation, normalization, PCA, unmixing, color transforms.",
    icon="icons/PixelTransform.png",  # Reuse manipulator icon or create new one
    after="imageio",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/pixel/"  # Reuse manipulator icons or create new ones


############################################
# Sensor band mapping for predefined indices (NIR, Red, Green, Blue, SWIR1, SWIR2)
############################################
_SENSOR_BANDS = {
    "Sentinel-2": {
        "red": "B4",
        "nir": "B8",
        "green": "B3",
        "blue": "B2",
        "swir1": "B11",
        "swir2": "B12",
    },
    "Landsat 8": {
        "red": "B4",
        "nir": "B5",
        "green": "B3",
        "blue": "B2",
        "swir1": "B6",
        "swir2": "B7",
    },
    "Landsat 5/7": {
        "red": "B3",
        "nir": "B4",
        "green": "B2",
        "blue": "B1",
        "swir1": "B5",
        "swir2": "B7",
    },
    "MODIS": {
        "red": "B1",
        "nir": "B2",
        "green": "B4",
        "blue": "B3",
        "swir1": "B6",
        "swir2": "B7",
    },
}


def _get_bands_for_sensor(sensor, custom_bands):
    """Resolve band name dict for the given sensor or custom bands."""
    if sensor == "Custom" and custom_bands:
        return custom_bands
    return _SENSOR_BANDS.get(sensor, _SENSOR_BANDS["Sentinel-2"])


############################################
# GEE Image Band Calculator
############################################


@knext.node(
    name="GEE Image Band Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandCalculator.png",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with calculated index/expression result.",
    port_type=gee_image_port_type,
)
class RasterCalculatorIndices:
    """Calculates predefined indices or custom expressions from image bands.

    This node computes indices or expressions using Calculation mode, Sensor, and Output band
    name parameters, and is commonly used to derive vegetation, water, and burn indices.

    This node provides both pre-defined indices and custom expression calculation capabilities.
    It is essential for vegetation monitoring, water body detection, burn severity assessment,
    and other remote sensing applications.

    **Pre-defined Indices:**

    - **NDVI**: Normalized Difference Vegetation Index - vegetation health
    - **NDWI**: Normalized Difference Water Index - water bodies
    - **NBR**: Normalized Burn Ratio - burn severity
    - **NDSI**: Normalized Difference Snow Index - snow/ice
    - **SAVI**: Soil Adjusted Vegetation Index - vegetation with soil correction
    - **EVI**: Enhanced Vegetation Index - improved vegetation index
    - **NDMI**: Normalized Difference Moisture Index - vegetation moisture
    - **GCI**: Green Chlorophyll Index - chlorophyll content

    **Custom Expressions (Calculation mode = Custom):**

    Expressions use **Earth Engine expression syntax**. Band names in the expression must match
    the names in the input image (e.g. ``B8``, ``B4``, ``elevation``, ``SR_B5``). Use the band
    name directly as a variable, or ``b(0)`` / ``b("band_name")`` to refer to bands by index or name.

    **Common symbols and operators:**

    - **Arithmetic:** ``+`` ``-`` ``*`` ``/`` ``%`` ``**``
    - **Comparison:** ``<`` ``<=`` ``>`` ``>=`` ``==`` ``!=``
    - **Logical:** ``&&`` (and), ``||`` (or), ``!`` (not)
    - **Conditional (reclassification):** ``?`` and ``:`` — ternary operator: ``condition ? valueIfTrue : valueIfFalse``.
      Chain for multiple classes: ``(a <= 100) ? 0 : (a <= 200) ? 1 : 2``
    - **Functions:** ``abs()``, ``sqrt()``, ``pow(x,y)``, ``exp()``, ``log()``, ``min()``, ``max()``, ``round()``, ``floor()``, ``ceil()``

    **Example expressions:**

    - **Vegetation index (NDVI):** ``(B8 - B4) / (B8 + B4)`` — use band names that match your image (e.g. B5/B4 for Landsat 8).
    - **Reclassify elevation into zones (e.g. GMTED band ``elevation``):**
      ``(elevation <= 100) ? 0 : (elevation <= 200) ? 1 : (elevation <= 500) ? 2 : 3``
      → output band name e.g. ``zone``. Replace ``elevation`` if your band has a different name (e.g. ``B1``).
    - **Binary mask (land/water):** ``elevation > 0 ? 1 : 0``
    - **Multi-threshold (e.g. loss year groups):** ``(lossyear >= 1 && lossyear <= 7) ? 1 : (lossyear <= 14) ? 2 : (lossyear <= 20) ? 3 : 0``

    Band names must be valid identifiers (letter or underscore first); for names with hyphens or
    leading digits, use ``b("band_name")`` in the expression.

    **Common Use Cases:**

    - Monitor vegetation health and phenology
    - Detect water bodies and wetlands
    - Assess fire damage and recovery
    - Map snow and ice coverage
    - Calculate custom spectral ratios
    - Create composite indices for classification

    **Sensor / band mapping:**

    When using a pre-defined index, select **Sensor** so the correct band names are used:
    - **Sentinel-2**: B4 (Red), B8 (NIR), B3 (Green), B2 (Blue), B11/B12 (SWIR)
    - **Landsat 8**: B4 (Red), B5 (NIR), B3 (Green), B2 (Blue), B6/B7 (SWIR)
    - **Landsat 5/7**: B3 (Red), B4 (NIR), B2 (Green), B1 (Blue), B5/B7 (SWIR)
    - **MODIS**: B1 (Red), B2 (NIR), B4 (Green), B3 (Blue), B6/B7 (SWIR)

    For other band names use **Calculation mode = Custom** and type an expression (e.g. ``(B5 - B4) / (B5 + B4)``).
    """

    calculation_mode = knext.StringParameter(
        "Calculation mode",
        "Choose between pre-defined indices or custom expression",
        default_value="predefined",
        enum=["predefined", "custom"],
    )

    predefined_index = knext.EnumParameter(
        "Pre-defined index",
        "Select a pre-defined vegetation or water index",
        default_value=knut.PredefinedIndex.NDVI.name,
        enum=knut.PredefinedIndex,
    ).rule(knext.OneOf(calculation_mode, ["predefined"]), knext.Effect.SHOW)

    sensor = knext.StringParameter(
        "Sensor / data source",
        "Select the satellite so the correct band names are used (e.g. Landsat 8 uses B5/B4 for NDVI). For other band names use Calculation mode = Custom and type an expression.",
        default_value="Sentinel-2",
        enum=["Sentinel-2", "Landsat 8", "Landsat 5/7", "MODIS"],
    ).rule(knext.OneOf(calculation_mode, ["predefined"]), knext.Effect.SHOW)

    custom_expression = knext.StringParameter(
        "Custom expression",
        "Earth Engine expression using band names as variables (e.g. (B8 - B4) / (B8 + B4)). "
        "Use condition ? valueIfTrue : valueIfFalse for reclassification. See node description for "
        "symbol reference and examples (NDVI, elevation zones, binary mask, multi-threshold).",
        default_value="(B8 - B4) / (B8 + B4)",
    ).rule(knext.OneOf(calculation_mode, ["custom"]), knext.Effect.SHOW)

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name for the calculated index band",
        default_value="index",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(self, exec_context: knext.ExecutionContext, image_connection):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image from connection
            image = image_connection.image

            if self.calculation_mode == "predefined":
                bands = _get_bands_for_sensor(self.sensor, None)
                result_image = self._calculate_predefined_index(
                    image, self.predefined_index, bands
                )
                band_name = self.predefined_index.lower()  # .name e.g. NDVI -> ndvi
            else:
                # Calculate custom expression
                result_image = self._calculate_custom_expression(
                    image, self.custom_expression
                )
                band_name = self.output_band_name

            # Add the calculated band to the image
            if result_image is not None:
                # Propagate input image mask so clipped/masked areas stay no-data (e.g. after Clip node)
                if isinstance(result_image, ee.Image):
                    result_image = result_image.updateMask(image.select(0).mask())
                # If result is a single band, add it to the original image
                if isinstance(result_image, ee.Image):
                    final_image = image.addBands(result_image.rename(band_name))
                else:
                    # If result is already a multi-band image
                    final_image = result_image
            else:
                LOGGER.error("Failed to calculate index/expression")
                return knut.export_gee_image_connection(image, image_connection)

            LOGGER.warning(f"Successfully calculated {band_name} index/expression")
            return knut.export_gee_image_connection(final_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Index calculation failed: {e}")
            raise

    def _calculate_predefined_index(self, image, index_name, bands):
        """Calculate pre-defined indices using sensor-specific band names.
        bands: dict with keys red, nir, green, blue, swir1, swir2 (string band names).
        """
        import ee

        r, nir, g, b, s1, s2 = (
            bands.get("red", "B4"),
            bands.get("nir", "B8"),
            bands.get("green", "B3"),
            bands.get("blue", "B2"),
            bands.get("swir1", "B11"),
            bands.get("swir2", "B12"),
        )

        if index_name == "NDVI":
            return image.normalizedDifference([nir, r]).rename("ndvi")
        elif index_name == "NDWI":
            return image.normalizedDifference([g, nir]).rename("ndwi")
        elif index_name == "NBR":
            return image.normalizedDifference([nir, s2]).rename("nbr")
        elif index_name == "NDSI":
            return image.normalizedDifference([g, s1]).rename("ndsi")
        elif index_name == "SAVI":
            return image.expression(
                "((NIR - Red) / (NIR + Red + 0.5)) * 1.5",
                {"NIR": image.select(nir), "Red": image.select(r)},
            ).rename("savi")
        elif index_name == "EVI":
            return image.expression(
                "2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))",
                {
                    "NIR": image.select(nir),
                    "Red": image.select(r),
                    "Blue": image.select(b),
                },
            ).rename("evi")
        elif index_name == "NDMI":
            return image.normalizedDifference([nir, s1]).rename("ndmi")
        elif index_name == "GCI":
            return image.expression(
                "(NIR / Green) - 1",
                {"NIR": image.select(nir), "Green": image.select(g)},
            ).rename("gci")
        else:
            return None

    def _calculate_custom_expression(self, image, expression):
        """Calculate custom expression with support for arbitrary band names"""
        import ee
        import re

        try:
            # Get all available band names from the image
            available_bands = image.bandNames().getInfo()

            # Extract potential variable names from expression
            # Match identifiers (letters, numbers, underscores) that are valid Python identifiers
            # This will match both B8, B4, B2 and nirScaled, redScaled, blueScaled
            potential_names = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression)

            # Filter to only include names that exist as bands in the image
            # Skip Python keywords and built-in functions
            python_keywords = {
                "and",
                "or",
                "not",
                "if",
                "else",
                "True",
                "False",
                "None",
                "abs",
                "sqrt",
                "pow",
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "min",
                "max",
                "round",
                "floor",
                "ceil",
                "int",
                "float",
            }

            band_dict = {}
            for name in potential_names:
                # Skip Python keywords and built-in functions
                if name in python_keywords:
                    continue
                # Check if this name exists as a band in the image
                if name in available_bands:
                    band_dict[name] = image.select(name)

            # Calculate expression
            result = image.expression(expression, band_dict)
            return result

        except Exception as e:
            import logging

            LOGGER = logging.getLogger(__name__)
            LOGGER.error(f"Custom expression calculation failed: {e}")
            return None


############################################
# GEE Image MultiBand Calculator
############################################


@knext.node(
    name="GEE Image MultiBand Calculator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "MultiCalculator.png",
    id="imagemultibandcalculator",
    after="imagevaluefilter",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image to apply the batch expression to.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image with new bands added from batch expression.",
    port_type=gee_image_port_type,
)
class ImageMultiBandCalculator:
    """Applies a template expression to multiple bands in one batch, same as IC MultiBand Calculator.

    This node applies a batch expression using Expression template, Rows, and Output band names parameters, and is commonly used to scale or transform many bands consistently.

    Use **BX1, BX2, BX3, BX4** as placeholders. Rows separated by **semicolon (;)**, within each row **comma-separated** (one row per output band).
    For property names use **prop(\"BX2\")**, **prop(\"BX3\")** (keep quotes around BX2/BX3).
    """

    expression = knext.StringParameter(
        "Expression template",
        'BX1 = band (no quotes); property placeholders in prop("BX2"), prop("BX3"). E.g. BX1 * prop("BX2") + prop("BX3") or BX1 * 0.0000275 - 0.2.',
        default_value='BX1 * prop("BX2") + prop("BX3")',
    )

    rows = knext.StringParameter(
        "Rows (one per output band)",
        "Rows separated by semicolon (;); within each row, comma-separated values for BX1, BX2, ...",
        default_value="SR_B1, REFLECTANCE_MULT_BAND_1, REFLECTANCE_ADD_BAND_1; SR_B2, REFLECTANCE_MULT_BAND_2, REFLECTANCE_ADD_BAND_2; SR_B3, REFLECTANCE_MULT_BAND_3, REFLECTANCE_ADD_BAND_3",
    )

    output_band_names = knext.StringParameter(
        "Output band names",
        "Comma-separated names for each row's output band. If empty, use first column (BX1) of each row.",
        default_value="",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import re
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)
        img = image_connection.image
        expression = (self.expression or "").strip()
        rows_str = (self.rows or "").strip()
        if not expression:
            raise ValueError("Expression template is required.")
        if not rows_str:
            raise ValueError("Rows are required.")

        bx_matches = list(re.finditer(r"BX(\d+)", expression, re.IGNORECASE))
        if not bx_matches:
            raise ValueError(
                "Expression must contain at least one placeholder BX1, BX2, ..."
            )
        n_cols = max(int(m.group(1)) for m in bx_matches)

        lines = [ln.strip() for ln in rows_str.split(";") if ln.strip()]
        row_list = []
        for ln in lines:
            vals = [v.strip() for v in ln.split(",") if v.strip()]
            if len(vals) < n_cols:
                raise ValueError(
                    f"Row has {len(vals)} values but expression needs {n_cols} (BX1..BX{n_cols})."
                )
            row_list.append(vals[:n_cols])

        out_names_str = (self.output_band_names or "").strip()
        if out_names_str:
            output_band_names = [
                n.strip() for n in out_names_str.split(",") if n.strip()
            ]
            if len(output_band_names) != len(row_list):
                raise ValueError("Output band names count must match number of rows.")
        else:
            output_band_names = [row[0] for row in row_list]

        prop_pattern = re.compile(r'prop\s*\(\s*["\']([^"\']+)["\']\s*\)')
        python_keywords = {
            "and",
            "or",
            "not",
            "if",
            "else",
            "True",
            "False",
            "None",
            "abs",
            "sqrt",
            "pow",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "min",
            "max",
            "round",
            "floor",
            "ceil",
            "int",
            "float",
        }

        def build_modified_expr_and_dict(substituted_expr, image):
            prop_to_placeholder = {}
            placeholders_ordered = []

            def repl(m):
                pname = m.group(1)
                if pname not in prop_to_placeholder:
                    prop_to_placeholder[pname] = f"_p{len(placeholders_ordered)}"
                    placeholders_ordered.append(pname)
                return prop_to_placeholder[pname]

            modified_expr = prop_pattern.sub(repl, substituted_expr)
            potential = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", modified_expr)
            band_names = [
                n
                for n in potential
                if n not in python_keywords and not re.match(r"_p\d+$", n)
            ]
            band_names = list(dict.fromkeys(band_names))
            band_dict = {}
            for b in band_names:
                band_dict[b] = image.select(b)
            for prop_name, placeholder in prop_to_placeholder.items():
                band_dict[placeholder] = ee.Image.constant(image.get(prop_name))
            return modified_expr, band_dict

        replace_order = sorted(range(1, n_cols + 1), reverse=True)
        current = img
        for row, out_name in zip(row_list, output_band_names):
            substituted = expression
            for i in replace_order:
                substituted = substituted.replace(f"BX{i}", row[i - 1])
            modified_expr, band_dict = build_modified_expr_and_dict(
                substituted, current
            )
            result_band = current.expression(modified_expr, band_dict)
            current = current.addBands(result_band.rename(out_name), overwrite=True)

        LOGGER.warning("GEE Image MultiBand Calculator: %s bands", len(row_list))
        return knut.export_gee_image_connection(current, image_connection)


############################################
# GEE Image Band Aggregator
############################################


@knext.node(
    name="GEE Image Band Aggregator",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandAggregator.png",
    id="bandreducer",
    after="spatialstatistics",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection. All bands are reduced to one band per pixel.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with a single band (reduced from all input bands).",
    port_type=gee_image_port_type,
)
class ImageBandAggregator:
    """Aggregates all bands of an image to a single band per pixel using a reducer.

    This node reduces bands using Reducer and Output band name parameters, and is commonly
    used to create single-band inputs for downstream analysis.

    Uses Earth Engine's ``image.reduce(reducer)``. For each pixel, the reducer
    is applied across all band values; the result is a one-band image. Useful for:
    - **Max**: e.g. NAIP 4-band max for Geary's C or other single-band inputs
    - **Mean**: average across bands (e.g. pan- band)
    - **Min / Median / Sum**: other per-pixel band aggregates
    - **StdDev / Variance**: variability across bands at each pixel
    - **First / Last**: first or last band value (by band order)
    - **Count**: number of bands (constant image, useful for masking)
    - **Product**: product of all band values
    """

    reducer = knext.EnumParameter(
        "Reducer",
        "Reducer to apply across bands at each pixel. Output is one band.",
        default_value=knut.BandReducer.MAX.name,
        enum=knut.BandReducer,
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name for the single output band (e.g. 'max_bands' or leave default).",
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

        # Keys from knut.BandReducer so UI and reducer stay in sync
        reducer_map = {
            "max": ee.Reducer.max(),
            "min": ee.Reducer.min(),
            "mean": ee.Reducer.mean(),
            "median": ee.Reducer.median(),
            "sum": ee.Reducer.sum(),
            "product": ee.Reducer.product(),
            "stddev": ee.Reducer.stdDev(),
            "variance": ee.Reducer.variance(),
            "first": ee.Reducer.first(),
            "last": ee.Reducer.last(),
            "count": ee.Reducer.count(),
        }

        try:
            image = image_connection.image
            red = reducer_map[self.reducer.lower()]
            result = image.reduce(red)
            # Propagate input image mask so clipped/masked areas stay no-data (e.g. after Clip node)
            result = result.updateMask(image.select(0).mask())

            if self.output_band_name and self.output_band_name.strip():
                result = result.rename(self.output_band_name.strip())

            LOGGER.warning(
                f"Band reducer applied: {self.reducer}, output band(s): {result.bandNames().getInfo()}"
            )
            return knut.export_gee_image_connection(result, image_connection)

        except Exception as e:
            LOGGER.error(f"Band reducer failed: {e}")
            raise


############################################
# Band-wise Normalization
############################################


@knext.node(
    name="GEE Image Band-wise Normalization",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "BandNorm.png",
    id="bandwisenormalization",
    after="bandreducer",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection. Each band is normalized independently.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with per-band normalized bands.",
    port_type=gee_image_port_type,
)
class BandwiseNormalization:
    """Normalizes each band independently using per-band statistics from reduceRegion.

    This node normalizes bands using Transform type, Bands to process, and scale parameters,
    and is commonly used for preprocessing before modeling.

    Uses ``image.reduceRegion(ee.Reducer.minMax()/mean/stdDev/percentile)`` to compute
    per-band statistics over the image, then applies a transform to each band. Useful for:
    - **Max**: band / max (0–1, e.g. for preprocessing)
    - **Min-max**: (band - min) / (max - min) → 0–1
    - **Z-score**: (band - mean) / stdDev
    - **Percentile**: band / percentile(band, p) — robust to outliers
    - **Clip**: band.clamp(low, high) — per-band or global bounds

    **Parameters:**
    - **Scale / maxPixels**: For reduceRegion (statistics over image).
    """

    transform_type = knext.StringParameter(
        "Transform type",
        "Per-band normalization method.",
        default_value="max",
        enum=["max", "min_max", "z_score", "percentile", "clip"],
    )

    bands = knext.StringParameter(
        "Bands to process",
        "Comma-separated band names (e.g. 'B8,B11,B12'). Leave empty for all bands.",
        default_value="",
    )

    percentile_value = knext.IntParameter(
        "Percentile (for percentile transform)",
        "Percentile to use as divisor (e.g. 99 for robust scaling).",
        default_value=99,
        min_value=1,
        max_value=100,
    ).rule(knext.OneOf(transform_type, ["percentile"]), knext.Effect.SHOW)

    clip_use_band_stats = knext.BoolParameter(
        "Use per-band min/max for clip bounds",
        "If enabled, uses reduceRegion min/max per band. If disabled, uses Clip low/high below.",
        default_value=True,
    ).rule(knext.OneOf(transform_type, ["clip"]), knext.Effect.SHOW)

    clip_low = knext.DoubleParameter(
        "Clip low",
        "Lower bound for clip transform (when not using per-band stats).",
        default_value=0.0,
    ).rule(
        knext.And(
            knext.OneOf(transform_type, ["clip"]),
            knext.OneOf(clip_use_band_stats, [False]),
        ),
        knext.Effect.SHOW,
    )

    clip_high = knext.DoubleParameter(
        "Clip high",
        "Upper bound for clip transform (when not using per-band stats).",
        default_value=1.0,
    ).rule(
        knext.And(
            knext.OneOf(transform_type, ["clip"]),
            knext.OneOf(clip_use_band_stats, [False]),
        ),
        knext.Effect.SHOW,
    )

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        max_value=1000,
        scale_description="Pixel scale for reduceRegion (statistics computation). Only used when Use NominalScale is disabled.",
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=10000000,
        description="Maximum pixels for reduceRegion (statistics computation).",
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

        try:
            image = image_connection.image
            band_names = image.bandNames().getInfo()
            if self.bands and self.bands.strip():
                to_process = [b.strip() for b in self.bands.split(",") if b.strip()]
                to_process = [b for b in to_process if b in band_names]
                if not to_process:
                    raise ValueError(
                        f"No valid bands found. Requested: {self.bands}, available: {band_names}"
                    )
            else:
                to_process = band_names

            img = image.select(to_process)
            geom = img.geometry()
            scale = knut.resolve_scale(self.use_nominal_scale, self.scale, image)
            max_px = self.max_pixels
            tt = self.transform_type

            result = None

            if tt == "max":
                stats = img.reduceRegion(
                    ee.Reducer.max(),
                    geometry=geom,
                    scale=scale,
                    maxPixels=max_px,
                    bestEffort=True,
                )
                band_max = stats.getInfo()
                for bn in to_process:
                    v = band_max.get(bn, 1)
                    if abs(float(v)) < 1e-10:
                        v = 1.0
                    band_img = img.select(bn).divide(float(v)).rename(bn)
                    if result is None:
                        result = band_img
                    else:
                        result = result.addBands(band_img)

            elif tt == "min_max":
                stats = img.reduceRegion(
                    ee.Reducer.minMax(),
                    geometry=geom,
                    scale=scale,
                    maxPixels=max_px,
                    bestEffort=True,
                )
                st = stats.getInfo()
                for bn in to_process:
                    mn = st.get(f"{bn}_min")
                    mx = st.get(f"{bn}_max")
                    if mn is not None and mx is not None:
                        rng = float(mx) - float(mn)
                        if abs(rng) < 1e-10:
                            rng = 1.0
                        band_img = (
                            img.select(bn).subtract(float(mn)).divide(rng).rename(bn)
                        )
                    else:
                        band_img = img.select(bn).multiply(0).rename(bn)
                    if result is None:
                        result = band_img
                    else:
                        result = result.addBands(band_img)

            elif tt == "z_score":
                stats = img.reduceRegion(
                    ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    geometry=geom,
                    scale=scale,
                    maxPixels=max_px,
                    bestEffort=True,
                )
                st = stats.getInfo()
                for bn in to_process:
                    mu = st.get(f"{bn}_mean")
                    sd = st.get(f"{bn}_stdDev")
                    if mu is not None and sd is not None and abs(float(sd)) >= 1e-10:
                        band_img = (
                            img.select(bn)
                            .subtract(float(mu))
                            .divide(float(sd))
                            .rename(bn)
                        )
                    else:
                        band_img = (
                            img.select(bn)
                            .subtract(float(mu) if mu is not None else 0)
                            .multiply(0)
                            .rename(bn)
                        )
                    if result is None:
                        result = band_img
                    else:
                        result = result.addBands(band_img)

            elif tt == "percentile":
                stats = img.reduceRegion(
                    ee.Reducer.percentile([self.percentile_value]),
                    geometry=geom,
                    scale=scale,
                    maxPixels=max_px,
                    bestEffort=True,
                )
                st = stats.getInfo()
                key_prefix = f"_p{self.percentile_value}"
                for bn in to_process:
                    key = f"{bn}{key_prefix}"
                    v = st.get(key)
                    if v is not None and abs(float(v)) >= 1e-10:
                        band_img = img.select(bn).divide(float(v)).rename(bn)
                    else:
                        band_img = img.select(bn).multiply(0).rename(bn)
                    if result is None:
                        result = band_img
                    else:
                        result = result.addBands(band_img)

            elif tt == "clip":
                if self.clip_use_band_stats:
                    stats = img.reduceRegion(
                        ee.Reducer.minMax(),
                        geometry=geom,
                        scale=scale,
                        maxPixels=max_px,
                        bestEffort=True,
                    )
                    st = stats.getInfo()
                for bn in to_process:
                    if self.clip_use_band_stats:
                        mn = st.get(f"{bn}_min")
                        mx = st.get(f"{bn}_max")
                        lo = float(mn) if mn is not None else 0.0
                        hi = float(mx) if mx is not None else 1.0
                    else:
                        lo = self.clip_low
                        hi = self.clip_high
                    band_img = img.select(bn).clamp(lo, hi).rename(bn)
                    if result is None:
                        result = band_img
                    else:
                        result = result.addBands(band_img)

            else:
                raise ValueError(f"Unknown transform type: {tt}")

            LOGGER.warning(f"Band-wise normalization: {tt}, bands={to_process}")
            return knut.export_gee_image_connection(result, image_connection)

        except Exception as e:
            LOGGER.error(f"Band-wise normalization failed: {e}")
            raise


############################################
# Terrain (Slope / Aspect / Hillshade)
############################################


class TerrainOutputOptions(knext.EnumParameterOptions):
    """Terrain derivatives to compute from elevation."""

    SLOPE = ("Slope", "Slope in degrees [0, 90)")
    ASPECT = ("Aspect", "Aspect in degrees (0=N, 90=E, 180=S, 270=W)")
    HILLSHADE = ("Hillshade", "Hillshade for visualization (0–255)")

    @classmethod
    def get_default(cls):
        return [cls.SLOPE.name]


@knext.node(
    name="Terrain from Elevation",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "TerrainFromElev.png",
    id="terrain",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="Elevation image (DEM) in meters: single band or a band named 'elevation'.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="Image with elevation plus selected terrain bands (slope, aspect, hillshade).",
    port_type=gee_image_port_type,
)
class TerrainDerivatives:
    """Computes terrain derivatives from a DEM: slope, aspect, and/or hillshade.

    This node derives terrain products using Elevation band plus output selection or Compute all
    products, and is commonly used to generate slope, aspect, and hillshade for analysis.

    **Compute all products:** If enabled, uses ``ee.Terrain.products(dem)`` to compute
    slope, aspect, and hillshade in one call (hillshade uses default azimuth=270°, elevation=45°).

    **Otherwise:** Select which derivatives to compute; slope and aspect use ``ee.Terrain.slope``
    and ``ee.Terrain.aspect``; hillshade uses ``ee.Terrain.hillshade(dem, azimuth, elevation)``
    with the Advanced hillshade parameters.
    """

    elevation_band = knext.StringParameter(
        "Elevation band",
        "Band name to use as DEM (e.g. 'elevation'). Required.",
        default_value="elevation",
    )

    use_products = knext.BoolParameter(
        "Compute all products",
        "If enabled, compute slope, aspect, and hillshade in one call (default illumination). If disabled, select outputs and set hillshade parameters.",
        default_value=False,
    )

    terrain_outputs = knext.EnumSetParameter(
        "Terrain outputs",
        "Select slope, aspect, and/or hillshade. Only used when 'Compute all products' is disabled.",
        default_value=TerrainOutputOptions.get_default(),
        enum=TerrainOutputOptions,
    ).rule(knext.OneOf(use_products, [False]), knext.Effect.SHOW)

    hillshade_azimuth = knext.IntParameter(
        "Hillshade azimuth (degrees)",
        "Illumination azimuth (0=N, 90=E, 180=S, 270=W). Used for hillshade when not using Compute all products.",
        default_value=270,
        min_value=0,
        max_value=360,
        is_advanced=True,
    )

    hillshade_elevation = knext.IntParameter(
        "Hillshade elevation (degrees)",
        "Illumination elevation (sun angle above horizon). Used for hillshade when not using Compute all products.",
        default_value=45,
        min_value=0,
        max_value=90,
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
        band_names = image.bandNames().getInfo()
        if not band_names:
            raise ValueError("Image has no bands.")

        elev_band = (self.elevation_band or "").strip()
        if not elev_band:
            raise ValueError("Elevation band is required.")
        if elev_band not in band_names:
            raise ValueError(
                f"Elevation band '{elev_band}' not in image bands: {band_names}"
            )
        dem = image.select(elev_band)

        if self.use_products:
            products_img = ee.Terrain.products(dem)
            # Keep all input bands and add slope, aspect, hillshade
            result = image.addBands(
                products_img.select(["slope", "aspect", "hillshade"])
            )
            LOGGER.warning("Terrain: computed slope, aspect, hillshade (products).")
        else:
            selected = [s.lower() for s in self.terrain_outputs]
            result = image
            if "slope" in selected:
                result = result.addBands(ee.Terrain.slope(dem))
            if "aspect" in selected:
                result = result.addBands(ee.Terrain.aspect(dem))
            if "hillshade" in selected:
                result = result.addBands(
                    ee.Terrain.hillshade(
                        dem,
                        self.hillshade_azimuth,
                        self.hillshade_elevation,
                    )
                )
            if not selected:
                raise ValueError(
                    "Select at least one terrain output when not using products."
                )
            LOGGER.warning(
                "Terrain: computed %s (azimuth=%s, elev=%s).",
                selected,
                self.hillshade_azimuth,
                self.hillshade_elevation,
            )

        return knut.export_gee_image_connection(result, image_connection)


############################################
# Tasseled Cap Transformation
############################################


@knext.node(
    name="GEE Tasseled Cap Transformation",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "TasseledCap.png",  # Placeholder
    id="tasseledcap",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with Landsat imagery (TOA reflectance).",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="GEE Array Connection (Coefficients)",
    description="GEE Array connection with TC transformation coefficients matrix (required). The array should be a 2D matrix where rows represent TC components and columns represent input bands. Use 'Table to GEE Array' node to create this connection.",
    port_type=gee_array_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with Tasseled Cap components (brightness, greenness, wetness, etc.).",
    port_type=gee_image_port_type,
)
class TasseledCapTransformation:
    """Applies Tasseled Cap transformation to Landsat imagery.

    This node applies TC transformation using Input bands and Output component names with
    a coefficients array, and is commonly used for vegetation and soil component analysis.

    The Tasseled Cap transformation is a linear transformation that rotates the
    original spectral space to maximize separation between different growth stages
    of crops. The output components are:
    - Brightness: Soil line component
    - Greenness: Vegetation component
    - Wetness: Moisture component
    - Additional components (4th, 5th, 6th)

    **Input Requirements:**

    - Landsat TOA (Top of Atmosphere) reflectance imagery
    - GEE Array connection with TC coefficients matrix (required): 2D array where rows = TC components, columns = input bands
    - Input bands parameter: Comma-separated list of band names matching the columns of the coefficient matrix

    **Output Components:**

    - brightness: Soil brightness component
    - greenness: Vegetation greenness component
    - wetness: Moisture/wetness component
    - fourth, fifth, sixth: Additional orthogonal components

    **Common Use Cases:**

    - Crop growth stage monitoring
    - Vegetation analysis
    - Soil brightness mapping
    - Moisture content assessment
    - Agricultural monitoring

    **Note:**
    - A custom coefficients array must be provided via the GEE Array Connection input port.
    - You must specify the input bands parameter to match the columns of the coefficient matrix.
    - The coefficient matrix should be a 2D array where rows = TC components, columns = input bands.
    """

    input_bands = knext.StringParameter(
        "Input bands",
        "Comma-separated list of input band names to use for TC transformation (e.g., 'B1,B2,B3,B4,B5,B7' for Landsat 5/7 or 'B2,B3,B4,B5,B6,B7' for Landsat 8). Must match the number of columns in the coefficient matrix.",
        default_value="",
    )

    component_names = knext.StringParameter(
        "Output component names",
        "Comma-separated list of component names for output bands (e.g., 'brightness,greenness,wetness,fourth,fifth,sixth'). Number of names must match the number of rows in the coefficient matrix (6 for Landsat).",
        default_value="brightness,greenness,wetness,fourth,fifth,sixth",
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        array_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            image = image_connection.image

            # Check if coefficients array is provided
            if array_connection is None or array_connection.array is None:
                raise ValueError(
                    "GEE Array Connection with coefficients matrix is required. "
                    "Please connect a Table to GEE Array node output containing the TC coefficients matrix."
                )

            # Use coefficients from input array connection
            tc_coefficients = array_connection.array

            # Parse input bands from parameter
            if not self.input_bands or not self.input_bands.strip():
                raise ValueError(
                    "Input bands parameter is required. "
                    "Please specify the input band names (e.g., 'B1,B2,B3,B4,B5,B7' for Landsat 5/7 or 'B2,B3,B4,B5,B6,B7' for Landsat 8)."
                )
            bands = [b.strip() for b in self.input_bands.split(",") if b.strip()]

            if not bands:
                raise ValueError("No valid input bands specified.")

            # Validate that number of bands matches coefficient matrix columns
            # Get array dimensions by converting to Python list
            coefficient_list = tc_coefficients.getInfo()
            num_coefficient_rows = len(coefficient_list)
            num_coefficient_cols = (
                len(coefficient_list[0]) if num_coefficient_rows > 0 else 0
            )

            if len(bands) != num_coefficient_cols:
                raise ValueError(
                    f"Number of input bands ({len(bands)}) must match "
                    f"number of coefficient matrix columns ({num_coefficient_cols})."
                )

            # Select required bands
            selected_image = image.select(bands)

            # Convert to array image (1D array per pixel)
            array_image_1d = selected_image.toArray()

            # Convert to 2D array (6x1 matrix per pixel) for matrix multiplication
            array_image_2d = array_image_1d.toArray(1)

            # Parse component names from parameter
            component_list = [
                comp.strip() for comp in self.component_names.split(",") if comp.strip()
            ]

            # Validate number of components matches coefficient matrix rows
            # num_coefficient_rows was already calculated above
            if len(component_list) != num_coefficient_rows:
                raise ValueError(
                    f"Number of component names ({len(component_list)}) must match "
                    f"number of coefficient matrix rows ({num_coefficient_rows}). "
                    f"Please provide exactly {num_coefficient_rows} component names."
                )

            # Apply matrix multiplication: TC = RT * p0
            tc_image = (
                ee.Image(tc_coefficients)
                .matrixMultiply(array_image_2d)
                .arrayProject([0])
                .arrayFlatten([component_list])
            )

            LOGGER.warning(
                f"Tasseled Cap transformation completed with components: {', '.join(component_list)}"
            )

            return knut.export_gee_image_connection(tc_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Tasseled Cap transformation failed: {e}")
            raise


############################################
# Principal Component Analysis
############################################


@knext.node(
    name="GEE Principal Component Analysis",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GeePCA.png",  # Placeholder
    id="pca",
    after="tasseledcap",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with multiple bands for PCA analysis. The image should be clipped to your study area before applying PCA. The covariance matrix will be calculated using the image's geometry boundary.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with principal component bands.",
    port_type=gee_image_port_type,
)
class PrincipalComponentAnalysis:
    """Performs Principal Component Analysis on multi-band imagery.

    This node performs PCA using Input bands, Output component names, and Max pixels parameters,
    and is commonly used for dimensionality reduction or feature extraction.

    PCA is an orthogonal linear transformation that transforms data into a new
    coordinate system where the first axis captures the largest variance, the second
    captures the second-largest variance, and so on. Principal components are uncorrelated.

    **Input Requirements:**

    - Multi-band image (typically 4-8 bands)
    - **Important:** The input image should be clipped to your study area before applying PCA.
      The covariance matrix will be calculated using the image's geometry boundary.

    **Output:**

    - Principal component bands (pca1, pca2, pca3, ...)
    - Number of components equals number of input bands (or as specified by component_names)

    **Common Use Cases:**

    - Dimensionality reduction
    - Noise reduction
    - Feature extraction for classification
    - Data compression
    - Change detection

    **Note:**
    - PCA components are ordered by variance (pca1 has highest variance).
    - Most information is typically in the first few components.
    - Use image clipping nodes before this node to limit the analysis area.
    """

    input_bands = knext.StringParameter(
        "Input bands",
        """Comma-separated list of band names to use for PCA (e.g., 'B2,B3,B4,B5,B6,B7').
        Leave empty to use all bands.""",
        default_value="",
    )

    component_names = knext.StringParameter(
        "Output component names",
        "Comma-separated list of component names for output bands (e.g., 'pca1,pca2,pca3'). Leave empty to output all components with default names (pca1, pca2, pca3, ...). Number of names determines how many components to output.",
        default_value="",
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=1000000000,
        description="Maximum number of pixels to use for covariance calculation. Use this to limit computation for very large images. If exceeded, GEE will use bestEffort mode.",
    )

    keep_original_bands = knext.BoolParameter(
        "Keep original bands",
        "If enabled, preserve original bands in output along with principal components (feature enhancement mode). If disabled, only output principal components (dimensionality reduction mode).",
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

        try:
            image = image_connection.image

            # Select input bands
            if self.input_bands:
                band_list = [b.strip() for b in self.input_bands.split(",")]
                pca_image = image.select(band_list)
            else:
                pca_image = image
                band_list = image.bandNames().getInfo()

            num_bands = len(band_list)
            LOGGER.warning(f"Performing PCA on {num_bands} bands: {band_list}")

            # Use image geometry for covariance calculation
            geometry = pca_image.geometry()

            # Convert to array image
            array_image = pca_image.toArray()

            # Calculate covariance matrix
            # Use image's native resolution (no scale specified)
            # bestEffort allows computation even if maxPixels is exceeded
            covar = array_image.reduceRegion(
                reducer=ee.Reducer.covariance(),
                geometry=geometry,
                maxPixels=self.max_pixels,
                bestEffort=True,
            )

            # Extract covariance matrix
            covar_array = ee.Array(covar.get("array"))

            # Compute eigenvectors and eigenvalues
            eigens = covar_array.eigen()

            # Extract eigenvectors (stored in the 0th position of the 1-axis)
            eigen_vectors = eigens.slice(1, 1)

            # Perform matrix multiplication: PC = eigenvectors * array_image
            principal_components = ee.Image(eigen_vectors).matrixMultiply(
                array_image.toArray(1)
            )

            # Parse component names from parameter
            if self.component_names and self.component_names.strip():
                # Use provided component names
                component_list = [
                    comp.strip()
                    for comp in self.component_names.split(",")
                    if comp.strip()
                ]
                num_output_components = len(component_list)

                # Validate number of components
                if num_output_components > num_bands:
                    raise ValueError(
                        f"Number of component names ({num_output_components}) cannot exceed "
                        f"number of input bands ({num_bands})."
                    )
            else:
                # Default: output all components with default names
                num_output_components = num_bands
                component_list = [f"pca{i+1}" for i in range(num_bands)]

            # Convert back to multi-band image with default names first
            default_pc_names = [f"pca{i+1}" for i in range(num_bands)]
            pc_image = principal_components.arrayProject([0]).arrayFlatten(
                [default_pc_names]
            )

            # Select requested number of components
            if num_output_components < num_bands:
                selected_pc_names = default_pc_names[:num_output_components]
                pc_image = pc_image.select(selected_pc_names)
            else:
                selected_pc_names = default_pc_names

            # Rename bands to user-specified names (if different from default)
            if component_list != selected_pc_names:
                pc_image = pc_image.rename(component_list)

            # Determine output image based on keep_original_bands setting
            if self.keep_original_bands:
                # Feature enhancement mode: add PCA bands to original image
                if self.input_bands:
                    # If specific bands were used for PCA, preserve other bands
                    all_original_bands = image.bandNames().getInfo()
                    pca_bands_set = set(band_list)
                    other_bands = [
                        b for b in all_original_bands if b not in pca_bands_set
                    ]

                    if other_bands:
                        # Add PCA to other bands
                        other_bands_image = image.select(other_bands)
                        output_image = other_bands_image.addBands(pc_image)
                        LOGGER.warning(
                            f"PCA completed: {num_bands} input bands -> {num_output_components} components. "
                            f"Preserved {len(other_bands)} original bands: {other_bands}"
                        )
                    else:
                        # All bands were used for PCA, add PCA to original
                        output_image = image.addBands(pc_image)
                        LOGGER.warning(
                            f"PCA completed: {num_bands} input bands -> {num_output_components} components. "
                            f"Added PCA components to original {num_bands} bands."
                        )
                else:
                    # All bands were used for PCA, add PCA to original
                    output_image = image.addBands(pc_image)
                    LOGGER.warning(
                        f"PCA completed: {num_bands} input bands -> {num_output_components} components. "
                        f"Added PCA components to original {num_bands} bands."
                    )
            else:
                # Dimensionality reduction mode: only output PCA components
                output_image = pc_image
                LOGGER.warning(
                    f"PCA completed: {num_bands} input bands -> {num_output_components} components: {', '.join(component_list)}"
                )

            return knut.export_gee_image_connection(output_image, image_connection)

        except Exception as e:
            LOGGER.error(f"PCA failed: {e}")
            raise


############################################
# Spectral Unmixing
############################################


@knext.node(
    name="GEE Spectral Unmixing",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "SpectralUnmix.png",  # Placeholder
    id="spectralunmixing",
    after="pca",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with multi-band imagery for unmixing.",
    port_type=gee_image_port_type,
)
@knext.input_table(
    name="Endmembers GeoTable",
    description="Local GeoTable containing endmember geometries (polygons or points) with label column.",
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with endmember fraction bands (band names from label column).",
    port_type=gee_image_port_type,
)
class SpectralUnmixing:
    """Performs linear spectral unmixing to estimate endmember fractions.

    This node estimates endmember fractions using Input bands and Label column parameters,
    and is commonly used for fractional cover or sub-pixel analysis.

    Spectral unmixing solves the equation: p = S * f, where:
    - p is the pixel spectrum (B bands)
    - S is the endmember matrix (B x P, where P is number of endmembers)
    - f is the endmember fraction vector (P x 1)

    **Input Requirements:**

    - Multi-band image
    - Local GeoTable with:
      - Geometry column (polygons or points)
      - Label column containing endmember class names
      - Multiple polygons with the same label will be dissolved automatically

    **Output:**

    - Endmember fraction bands (one band per unique label)
    - Band names are automatically derived from label column values
    - Fractions sum to 1.0 for each pixel

    **Common Use Cases:**

    - Forest degradation detection
    - Sub-pixel land cover mapping
    - Fractional cover estimation
    - Change detection

    **Note:** Endmembers should represent pure spectra (e.g., pure vegetation, pure soil, pure water).
    Multiple polygons with the same label are automatically dissolved before calculating mean spectra.
    """

    input_bands = knext.StringParameter(
        "Input bands",
        """Comma-separated list of band names to use for unmixing (e.g., 'B2,B3,B4,B5,B6,B7').
        Leave empty to use all bands.""",
        default_value="",
    )

    label_column = knext.ColumnParameter(
        "Label column",
        "Column containing endmember labels (e.g., 'label', 'class', 'type'). Output band names will be derived from unique values in this column.",
        port_index=1,  # 第二个输入端口（GeoTable）
    )

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        max_value=1000,
        scale_description="Pixel resolution in meters for calculating endmember spectra. Only used when Use NominalScale is disabled.",
    )

    def configure(self, configure_context, input_schema1, input_table_schema):
        # validate label column exists
        if input_table_schema is not None:
            self.label_column = knut.column_exists_or_preset(
                configure_context,
                self.label_column,
                input_table_schema,
                lambda col: True,  # accept any column type
            )
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        endmembers_table,  # GeoTable
    ):
        import ee
        import logging
        import geopandas as gp
        import geemap

        LOGGER = logging.getLogger(__name__)

        try:
            image = image_connection.image

            scale_value = knut.resolve_scale(self.use_nominal_scale, self.scale, image)

            # 1. convert GeoTable to GeoDataFrame
            LOGGER.warning("Converting GeoTable to GeoDataFrame")

            df = endmembers_table.to_pandas()

            # Remove <RowID> if present
            if "<RowID>" in df.columns:
                df = df.drop(columns=["<RowID>"])
            df = df.reset_index(drop=True)

            # find geometry column
            geo_col = None
            for col in df.columns:
                if col == self.label_column:
                    continue
                # check if column is geometry column
                if len(df) > 0 and hasattr(df[col].iloc[0], "__geo_interface__"):
                    geo_col = col
                    break

            if geo_col is None:
                # try common geometry column names
                for col_name in ["geometry", "geom", "geometries", "the_geom"]:
                    if col_name in df.columns:
                        geo_col = col_name
                        break

            if geo_col is None:
                raise ValueError(
                    "No geometry column found in GeoTable. "
                    "Please ensure your table contains a geometry column."
                )

            # create GeoDataFrame
            shp = gp.GeoDataFrame(df, geometry=geo_col)

            # ensure CRS is WGS84
            if shp.crs is None:
                shp.set_crs(epsg=4326, inplace=True)
            else:
                shp.to_crs(4326, inplace=True)

            LOGGER.warning(
                f"GeoDataFrame created with {len(shp)} features, CRS: {shp.crs}"
            )

            # 2. get unique label values (for output band names)
            if self.label_column not in shp.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in GeoTable. "
                    f"Available columns: {list(shp.columns)}"
                )

            # get unique label values and sort
            endmember_labels = sorted(shp[self.label_column].dropna().unique().tolist())

            if len(endmember_labels) == 0:
                raise ValueError(
                    f"No valid labels found in column '{self.label_column}'. "
                    "Please ensure the label column contains non-null values."
                )

            num_endmembers = len(endmember_labels)
            LOGGER.warning(
                f"Found {num_endmembers} unique endmembers: {endmember_labels}"
            )

            # 3. dissolve multiple polygons with the same label
            LOGGER.warning("Dissolving polygons with the same label...")

            # use dissolve to merge geometries by label column
            # this will ensure each label has only one feature
            dissolved_shp = shp.dissolve(by=self.label_column, as_index=False)

            # ensure order after dissolve is consistent with endmember_labels
            # create a dictionary to map label to index
            label_to_order = {label: idx for idx, label in enumerate(endmember_labels)}
            dissolved_shp["_sort_order"] = dissolved_shp[self.label_column].map(
                label_to_order
            )
            dissolved_shp = dissolved_shp.sort_values("_sort_order").drop(
                columns=["_sort_order"]
            )

            # validate all labels exist
            dissolved_labels = dissolved_shp[self.label_column].tolist()
            if set(dissolved_labels) != set(endmember_labels):
                missing = set(endmember_labels) - set(dissolved_labels)
                raise ValueError(
                    f"Some labels were lost during dissolve operation. "
                    f"Missing labels: {missing}"
                )

            LOGGER.warning(
                f"Dissolved to {len(dissolved_shp)} features (one per endmember label)"
            )

            # 4. convert to FeatureCollection
            LOGGER.warning("Converting dissolved GeoDataFrame to FeatureCollection")
            endmembers_fc = geemap.gdf_to_ee(dissolved_shp)

            # 5. continue with existing unmixing logic
            # Select input bands
            if self.input_bands:
                band_list = [b.strip() for b in self.input_bands.split(",")]
                unmix_image = image.select(band_list)
            else:
                unmix_image = image
                band_list = image.bandNames().getInfo()

            num_bands = len(band_list)
            LOGGER.warning(f"Spectral unmixing with {num_bands} bands: {band_list}")

            # 6. calculate mean spectrum for each endmember
            endmember_means = []
            for label in endmember_labels:
                # Filter features with this label (should only be one, because already dissolved)
                endmember_fc = endmembers_fc.filter(
                    ee.Filter.eq(self.label_column, label)
                )

                # calculate mean spectrum over all dissolved polygons for this label
                mean_dict = unmix_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=endmember_fc.geometry(),  # this will include all dissolved geometries
                    scale=scale_value,
                    bestEffort=True,
                )

                # get mean values as list (sorted by band name)
                mean_values = mean_dict.values(band_list)
                endmember_means.append(mean_values)

            # 7. build endmember matrix and solve
            # Stack endmember vectors into matrix (B x P)
            endmembers_array = ee.Array.cat(endmember_means, 1)

            # convert input image to array (6x1 matrix per pixel)
            array_image = unmix_image.toArray().toArray(1)

            # solve for fractions: f = S^(-1) * p (using matrixSolve)
            unmixed = ee.Image(endmembers_array).matrixSolve(array_image)

            # 8. convert back to multi-band image, using label values as band names
            unmixed_image = unmixed.arrayProject([0]).arrayFlatten([endmember_labels])

            LOGGER.warning(
                f"Spectral unmixing completed: {num_endmembers} endmember fractions calculated. "
                f"Output bands: {endmember_labels}"
            )

            return knut.export_gee_image_connection(unmixed_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Spectral unmixing failed: {e}")
            raise


############################################
# HSV Color Transform
############################################


@knext.node(
    name="GEE HSV Color Transform",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "HSVtransform.png",  # Placeholder
    id="hsvtransform",
    after="spectralunmixing",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with RGB bands for HSV conversion.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with HSV bands or converted back to RGB.",
    port_type=gee_image_port_type,
)
class HSVColorTransform:
    """Converts between RGB and HSV color spaces.

    This node converts color space using Transform direction and RGB band names parameters,
    and is commonly used for pan-sharpening or color manipulation.

    HSV (Hue, Saturation, Value) color space is useful for:
    - Pan-sharpening: Replace Value band with panchromatic band
    - Color manipulation
    - Image enhancement

    **Transform Directions:**

    **RGB to HSV:**
    - Converts RGB image to HSV color space
    - **Output bands**: Three bands named `hue`, `saturation`, `value`
    - Requires specifying RGB band names (or uses first 3 bands if empty)
    - Optional panchromatic band for pan-sharpening workflow

    **HSV to RGB:**
    - Converts HSV image back to RGB color space
    - **Output bands**: Three bands with default names `red`, `green`, `blue` (no need to specify)
    - Input must contain bands named `hue`, `saturation`, `value`
    - Automatically selects these bands from the input image

    **Input Requirements:**

    - **RGB to HSV**: RGB image with 3 bands (Red, Green, Blue)
    - **HSV to RGB**: HSV image with bands named `hue`, `saturation`, `value`
    - For pan-sharpening: Additional panchromatic band

    **Output:**

    - **RGB to HSV**: Three bands (`hue`, `saturation`, `value`)
    - **HSV to RGB**: Three bands (`red`, `green`, `blue` - default names)

    **Common Use Cases:**

    - Pan-sharpening (combine high-res panchromatic with low-res RGB)
    - Color space conversion for processing
    - Image enhancement

    **Pan-sharpening Workflow:**

    1. RGB to HSV: Convert RGB bands to HSV (outputs: hue, saturation, value)
    2. Replace Value: Use panchromatic band instead of value band
    3. HSV to RGB: Convert back to RGB (outputs: red, green, blue)

    **Visualization:**

    - For HSV output: Use `GEE Image View` with bands: `hue,saturation,value`
    - For RGB output: Use `GEE Image View` with bands: `red,green,blue` (or leave empty for auto)
    """

    transform_direction = knext.StringParameter(
        "Transform direction",
        "Direction of color space conversion.",
        default_value="RGB to HSV",
        enum=["RGB to HSV", "HSV to RGB"],
    )

    rgb_bands = knext.StringParameter(
        "RGB band names",
        """Comma-separated list of RGB band names in order (e.g., 'B4,B3,B2' for Landsat).
        Leave empty to use first 3 bands.
        Only used when Transform direction is 'RGB to HSV'.""",
        default_value="",
    ).rule(knext.OneOf(transform_direction, ["RGB to HSV"]), knext.Effect.SHOW)

    pan_band = knext.StringParameter(
        "Panchromatic band (for pan-sharpening)",
        """Name of panchromatic band to replace Value band in HSV (optional).
        Only used when Transform direction is 'RGB to HSV' for pan-sharpening workflow.
        If provided, the node will: RGB→HSV→replace Value with pan band→RGB.
        The final output will be RGB bands (red, green, blue) instead of HSV bands.""",
        default_value="",
    ).rule(knext.OneOf(transform_direction, ["HSV to RGB"]), knext.Effect.SHOW)

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

        try:
            image = image_connection.image

            if self.transform_direction == "RGB to HSV":
                # Select RGB bands
                if self.rgb_bands:
                    rgb_band_list = [b.strip() for b in self.rgb_bands.split(",")]
                    if len(rgb_band_list) != 3:
                        raise ValueError("RGB bands must contain exactly 3 band names")
                    rgb_image = image.select(rgb_band_list)
                else:
                    # Use first 3 bands
                    rgb_image = image.select([0, 1, 2])

                # Convert RGB to HSV
                hsv_image = rgb_image.rgbToHsv()

                # If pan band is provided, replace value band with pan band
                if self.pan_band:
                    pan_band_image = image.select(self.pan_band)
                    # HSV bands are: hue, saturation, value
                    hsv_image = ee.Image.cat(
                        [
                            hsv_image.select("hue"),
                            hsv_image.select("saturation"),
                            pan_band_image,
                        ]
                    ).hsvToRgb()

                    LOGGER.warning(
                        f"Pan-sharpening applied: replaced Value with {self.pan_band} band"
                    )
                else:
                    LOGGER.warning("RGB to HSV conversion completed")

                return knut.export_gee_image_connection(hsv_image, image_connection)

            else:  # HSV to RGB
                # Select HSV bands
                hsv_image = image.select(["hue", "saturation", "value"])

                # Convert HSV to RGB
                rgb_image = hsv_image.hsvToRgb()

                LOGGER.warning("HSV to RGB conversion completed")

                return knut.export_gee_image_connection(rgb_image, image_connection)

        except Exception as e:
            LOGGER.error(f"HSV color transform failed: {e}")
            raise


############################################
# Table to GEE Array
############################################


@knext.node(
    name="Table to GEE Array",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "TableToArray.png",
    id="tabletoarray",
    after="hsvtransform",
)
@knext.input_table(
    name="Input Table",
    description="Table containing numeric data to convert to GEE Array. Each row becomes a row in the array, each column becomes a column.",
)
@knext.input_port(
    name="Google Earth Engine Connection",
    description="Google Earth Engine connection from the GEE Connector node.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Array Connection",
    description="GEE Array connection with embedded array object.",
    port_type=gee_array_port_type,
)
class TableToArray:
    """Converts a KNIME table to a Google Earth Engine Array.

    This node converts numeric tables into a GEE array with no parameters and is
    commonly used to provide coefficient matrices for PCA or Tasseled Cap.

    This node converts numeric table data into an ee.Array object,
    which can be used for matrix operations like Tasseled Cap transformation,
    PCA, spectral unmixing, etc.

    **Input Requirements:**

    - Table with numeric columns
    - Each row becomes a row in the array
    - Each column becomes a column in the array
    - Non-numeric columns are ignored

    **Output:**

    - GEE Array connection object
    - Array dimensions: [num_rows x num_columns]

    **Common Use Cases:**

    - **Tasseled Cap coefficients**: Upload TC transformation matrices
    - **PCA eigenvectors**: Upload principal component vectors
    - **Spectral unmixing endmembers**: Upload endmember spectra
    - **Custom transformation matrices**: Any matrix-based transformation

    **Example:**

    For Tasseled Cap Landsat 5 coefficients:
    - Table with 6 rows (components) and 6 columns (bands)
    - Each row represents one TC component (brightness, greenness, etc.)
    - Each column represents one Landsat band (B1, B2, B3, B4, B5, B7)

    **Note:** The array is created from all numeric columns in the table.
    Use column filtering or table manipulation nodes to prepare your data.
    """

    def configure(self, configure_context, input_table_schema, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        input_table: knext.Table,
        gee_connection,
    ):
        import ee
        import logging
        import pandas as pd
        import numpy as np

        LOGGER = logging.getLogger(__name__)

        try:
            # Convert table to pandas DataFrame
            df = input_table.to_pandas()

            # Remove RowID column if present
            if "<RowID>" in df.columns:
                df = df.drop(columns=["<RowID>"])

            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                raise ValueError(
                    "No numeric columns found in input table. Please ensure your table contains numeric data."
                )

            # Convert to list of lists (rows)
            # Each row in the table becomes a row in the array
            array_data = numeric_df.values.tolist()

            num_rows = len(array_data)
            num_cols = len(array_data[0]) if array_data else 0

            LOGGER.warning(
                f"Converting table to GEE Array: {num_rows} rows x {num_cols} columns"
            )

            # Create ee.Array
            gee_array = ee.Array(array_data)

            # Create connection object using helper function
            array_connection = knut.export_gee_array_connection(
                gee_array, gee_connection
            )

            LOGGER.warning(
                f"Successfully created GEE Array with shape [{num_rows} x {num_cols}]"
            )

            return array_connection

        except Exception as e:
            LOGGER.error(f"Table to Array conversion failed: {e}")
            raise
