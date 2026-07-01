"""
GEE Neighborhood nodes for KNIME.
Neighborhood and kernel operations: convolution, focal reduce, GLCM texture, morphology.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
)

# Category for GEE Neighborhood Transformation nodes
__category = knext.category(
    path="/community/gee",
    level_id="focal",
    name="GEE Neighborhood Transformation",
    description="Neighborhood and kernel operations: convolution, focal reduce, GLCM texture, morphology.",
    icon="icons/focal.png",  # Reuse tool icon or create new one
    after="pixeltransform",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/focal/"  # Reuse tool icons or create new ones


class KernelTypeOptions(knext.EnumParameterOptions):
    """Options for Image Convolution kernel type."""

    UNIFORM = ("Uniform", "Uniform square or circular kernel")
    GAUSSIAN = ("Gaussian", "Gaussian weighted kernel")
    LAPLACIAN = ("Laplacian", "Laplacian edge detection (4- or 8-connected)")
    SOBEL = ("Sobel", "Sobel edge detection")
    PREWITT = ("Prewitt", "Prewitt edge detection")
    ROBERTS = ("Roberts", "Roberts cross edge detection")
    CANNY = ("Canny", "Canny edge detection (Laplacian approximation)")
    CUSTOM = ("Custom", "Custom kernel weights")


class NeighborhoodReducerOptions(knext.EnumParameterOptions):
    """Options for Neighborhood Reducer statistic."""

    MEDIAN = ("Median", "Median value in neighborhood")
    MODE = ("Mode", "Most frequent value")
    MIN = ("Min", "Minimum value")
    MAX = ("Max", "Maximum value")
    MEAN = ("Mean", "Mean value")
    STDDEV = ("StdDev", "Standard deviation")
    VARIANCE = ("Variance", "Variance")
    SUM = ("Sum", "Sum of values")
    ENTROPY = ("Entropy", "Entropy (requires integer input)")


############################################
# Image Convolution
############################################


@knext.node(
    name="GEE Image Convolution",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImgConvolution.png",
    id="imageconvolution",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection for convolution operation.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with convolution result.",
    port_type=gee_image_port_type,
)
class ImageConvolution:
    """Performs linear convolution on images using various kernel types.

    This node applies convolution using Kernel type and related parameters to control
    kernel shape and size, and is commonly used for smoothing, edge detection, and feature extraction.

    **Kernel Types:**

    - **Uniform**: Square kernel with uniform weights (smoothing)
    - **Gaussian**: Gaussian-weighted kernel (smoothing with edge preservation)
    - **Laplacian**: Edge detection kernel
    - **Sobel**: Edge detection (horizontal and vertical)
    - **Prewitt**: Edge detection (horizontal and vertical)
    - **Roberts**: Edge detection (diagonal)
    - **Canny**: Multi-directional edge detection
    - **Custom**: User-defined kernel weights

    **Common Use Cases:**

    - Image smoothing and noise reduction
    - Edge detection and enhancement
    - Feature extraction
    - Preprocessing for classification

    **Note:** This node extends the existing **Focal Operations** node with more kernel types
    and convolution capabilities. Can replace **Focal Operations** for advanced use cases.
    """

    kernel_type = knext.EnumParameter(
        "Kernel type",
        "Type of convolution kernel to apply.",
        default_value=KernelTypeOptions.UNIFORM.name,
        enum=KernelTypeOptions,
    )

    kernel_radius = knext.DoubleParameter(
        "Kernel radius",
        "Radius of the kernel in pixels or meters. Only used for Uniform and Gaussian kernels.",
        default_value=2.0,
        min_value=0.5,
        max_value=50.0,
    ).rule(
        knext.OneOf(
            kernel_type,
            [KernelTypeOptions.UNIFORM.name, KernelTypeOptions.GAUSSIAN.name],
        ),
        knext.Effect.SHOW,
    )

    radius_units = knext.StringParameter(
        "Radius units",
        "Units for kernel radius. Only used for Uniform and Gaussian kernels.",
        default_value="meters",
        enum=["pixels", "meters"],
    ).rule(
        knext.OneOf(
            kernel_type,
            [KernelTypeOptions.UNIFORM.name, KernelTypeOptions.GAUSSIAN.name],
        ),
        knext.Effect.SHOW,
    )

    gaussian_sigma = knext.DoubleParameter(
        "Gaussian sigma",
        "Standard deviation for Gaussian kernel (only for Gaussian type).",
        default_value=1.0,
        min_value=0.1,
        max_value=10.0,
    ).rule(
        knext.OneOf(kernel_type, [KernelTypeOptions.GAUSSIAN.name]), knext.Effect.SHOW
    )

    gaussian_normalize = knext.BoolParameter(
        "Gaussian normalize",
        "If true, normalize kernel values to sum to 1. Advanced option.",
        default_value=True,
        is_advanced=True,
    ).rule(
        knext.OneOf(kernel_type, [KernelTypeOptions.GAUSSIAN.name]), knext.Effect.SHOW
    )

    gaussian_magnitude = knext.DoubleParameter(
        "Gaussian magnitude",
        "Scale each kernel value by this amount (e.g. -1 to invert). Default 1. Advanced option.",
        default_value=1.0,
        is_advanced=True,
    ).rule(
        knext.OneOf(kernel_type, [KernelTypeOptions.GAUSSIAN.name]), knext.Effect.SHOW
    )

    laplacian_type = knext.StringParameter(
        "Laplacian type",
        "Type of Laplacian kernel.",
        default_value="8-connected",
        enum=["4-connected", "8-connected"],
    ).rule(
        knext.OneOf(kernel_type, [KernelTypeOptions.LAPLACIAN.name]), knext.Effect.SHOW
    )

    sobel_direction = knext.StringParameter(
        "Sobel direction",
        "Direction of Sobel edge detection.",
        default_value="both",
        enum=["horizontal", "vertical", "both"],
    ).rule(knext.OneOf(kernel_type, [KernelTypeOptions.SOBEL.name]), knext.Effect.SHOW)

    prewitt_direction = knext.StringParameter(
        "Prewitt direction",
        "Direction of Prewitt edge detection.",
        default_value="both",
        enum=["horizontal", "vertical", "both"],
    ).rule(
        knext.OneOf(kernel_type, [KernelTypeOptions.PREWITT.name]), knext.Effect.SHOW
    )

    custom_kernel_weights = knext.StringParameter(
        "Custom kernel weights",
        """Custom kernel weights as comma-separated values (row by row).
        Example for 3x3: '1,1,1,1,0,1,1,1,1' (center is 0).
        Number of values must be (2*radius+1)^2.""",
        default_value="",
    ).rule(knext.OneOf(kernel_type, [KernelTypeOptions.CUSTOM.name]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
    ):
        import ee
        import logging
        import math

        LOGGER = logging.getLogger(__name__)

        try:
            image = image_connection.image

            # Create kernel based on type
            radius_units_str = self.radius_units  # "pixels" or "meters"
            if self.kernel_type == KernelTypeOptions.UNIFORM.name:
                kernel = ee.Kernel.square(self.kernel_radius, radius_units_str)

            elif self.kernel_type == KernelTypeOptions.GAUSSIAN.name:
                kernel = ee.Kernel.gaussian(
                    radius=self.kernel_radius,
                    sigma=self.gaussian_sigma,
                    units=radius_units_str,
                    normalize=self.gaussian_normalize,
                    magnitude=self.gaussian_magnitude,
                )

            elif self.kernel_type == KernelTypeOptions.LAPLACIAN.name:
                if self.laplacian_type == "8-connected":
                    kernel = ee.Kernel.laplacian8()
                else:
                    kernel = ee.Kernel.laplacian4()

            elif self.kernel_type == KernelTypeOptions.SOBEL.name:
                if self.sobel_direction == "horizontal":
                    kernel = ee.Kernel.sobel()
                elif self.sobel_direction == "vertical":
                    # Vertical Sobel is transpose of horizontal
                    kernel = ee.Kernel.sobel().transpose()
                else:  # BOTH
                    kernel = ee.Kernel.sobel()

            elif self.kernel_type == KernelTypeOptions.PREWITT.name:
                # Prewitt kernels (approximate with custom or use similar to Sobel)
                if self.prewitt_direction == "horizontal":
                    # Horizontal Prewitt: [-1,0,1; -1,0,1; -1,0,1]
                    weights = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                    kernel = ee.Kernel.fixed(3, 3, weights, -1, -1, False)
                elif self.prewitt_direction == "vertical":
                    # Vertical Prewitt: [-1,-1,-1; 0,0,0; 1,1,1]
                    weights = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
                    kernel = ee.Kernel.fixed(3, 3, weights, -1, -1, False)
                else:  # BOTH
                    weights = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                    kernel = ee.Kernel.fixed(3, 3, weights, -1, -1, False)

            elif self.kernel_type == KernelTypeOptions.ROBERTS.name:
                # Roberts cross kernel (2x2)
                weights = [1, 0, 0, -1]
                kernel = ee.Kernel.fixed(2, 2, weights, -1, -1, False)

            elif self.kernel_type == KernelTypeOptions.CANNY.name:
                # Canny edge detection uses multiple kernels
                # Simplified: use Laplacian as approximation
                kernel = ee.Kernel.laplacian8()

            elif self.kernel_type == KernelTypeOptions.CUSTOM.name:
                # Parse custom kernel weights
                if not self.custom_kernel_weights:
                    raise ValueError(
                        "Custom kernel weights must be provided for custom kernel type"
                    )

                weights = [
                    float(w.strip()) for w in self.custom_kernel_weights.split(",")
                ]
                size = int(math.sqrt(len(weights)))
                if size * size != len(weights):
                    raise ValueError(
                        f"Number of weights ({len(weights)}) must be a perfect square (e.g., 9 for 3x3, 25 for 5x5)"
                    )

                # Center offset
                center = size // 2
                kernel = ee.Kernel.fixed(size, size, weights, -center, -center, False)

            else:
                kernel = ee.Kernel.square(self.kernel_radius, radius_units_str)

            # Apply convolution
            convolved_image = image.convolve(kernel)

            LOGGER.warning(
                f"Convolution applied: kernel={self.kernel_type}, radius={self.kernel_radius} {self.radius_units}"
            )

            return knut.export_gee_image_connection(convolved_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Image convolution failed: {e}")
            raise


############################################
# Neighborhood Reducer (reduceNeighborhood)
############################################


@knext.node(
    name="GEE Neighborhood Reducer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "NeighborReducer.png",
    id="neighborhoodreducer",
    after="imageconvolution",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection for neighborhood reduction.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with reduced result.",
    port_type=gee_image_port_type,
)
class NeighborhoodReducer:
    """Applies a reducer over each pixel's neighborhood as a nonlinear focal operation.

    This node applies a neighborhood statistic using Reducer, Kernel radius, and Kernel
    shape parameters, and is commonly used for smoothing, morphology, or local texture measures.

    Uses Earth Engine's ``reduceNeighborhood`` (or ``entropy`` for that option): for each
    focal pixel, the chosen statistic is applied over the kernel window. Covers 10.2.2
    (median, mode), 10.2.3 (min/max morphology), and 10.2.4 texture (stdDev, variance, entropy).

    **Reducer options:**

    - **Median**: Denoising, edge-preserving smoothing (10.2.2).
    - **Mode**: Most frequent value; for categorical images.
    - **Min** / **Max**: Erosion / dilation in morphology.
    - **Mean**: Average in neighborhood (smoothing).
    - **StdDev**: Standard deviation in neighborhood .
    - **Variance**: Variance in neighborhood.
    - **Sum**: Sum of values in neighborhood.
    - **Entropy**: Entropy in neighborhood .
    """

    reducer_type = knext.EnumParameter(
        "Reducer",
        "Reducer or statistic to apply over each pixel's neighborhood.",
        default_value=NeighborhoodReducerOptions.MEDIAN.name,
        enum=NeighborhoodReducerOptions,
    )

    kernel_radius = knext.DoubleParameter(
        "Kernel radius",
        "Radius of the neighborhood kernel in pixels or meters.",
        default_value=2.0,
        min_value=0.5,
        max_value=50.0,
    )

    radius_units = knext.StringParameter(
        "Radius units",
        "Units for kernel radius.",
        default_value="meters",
        enum=["pixels", "meters"],
    )

    kernel_shape = knext.StringParameter(
        "Kernel shape",
        "Shape of the neighborhood kernel (structuring element).",
        default_value="square",
        enum=["square", "circle"],
    )

    keep_original_band_names = knext.BoolParameter(
        "Keep original band names",
        "If true, output bands keep the same names as the input image (recommended for chaining). If false, use GEE default names (e.g. band_median).",
        default_value=True,
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

            radius_units_str = self.radius_units  # "pixels" or "meters"
            if self.kernel_shape == "circle":
                kernel = ee.Kernel.circle(self.kernel_radius, radius_units_str)
            else:
                kernel = ee.Kernel.square(self.kernel_radius, radius_units_str)

            if self.reducer_type == NeighborhoodReducerOptions.ENTROPY.name:
                # Entropy uses image.entropy(kernel), requires integer input
                int_image = image.int()
                result = int_image.entropy(kernel)
            else:
                reducer_map = {
                    "median": ee.Reducer.median(),
                    "mode": ee.Reducer.mode(),
                    "min": ee.Reducer.min(),
                    "max": ee.Reducer.max(),
                    "mean": ee.Reducer.mean(),
                    "stddev": ee.Reducer.stdDev(),
                    "variance": ee.Reducer.variance(),
                    "sum": ee.Reducer.sum(),
                }
                reducer = reducer_map[self.reducer_type.lower()]
                result = image.reduceNeighborhood(reducer=reducer, kernel=kernel)

            if self.keep_original_band_names:
                result = result.rename(image.bandNames())

            LOGGER.warning(
                f"Neighborhood reducer applied: {self.reducer_type}, "
                f"kernel={self.kernel_shape} radius={self.kernel_radius} "
                f"{self.radius_units}"
            )

            return knut.export_gee_image_connection(result, image_connection)

        except Exception as e:
            LOGGER.error(f"Neighborhood reducer failed: {e}")
            raise


############################################
# GLCM Texture
############################################


@knext.node(
    name="GEE GLCM Texture",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "GLCM.png",
    id="glcmtexture",
    after="neighborhoodreducer",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection. Applied to all bands; image is cast to int internally.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with 18 GLCM texture bands per input band.",
    port_type=gee_image_port_type,
)
class GLCMTexture:
    """Computes texture metrics from the Gray Level Co-occurrence Matrix.

    This node computes texture features using Size, Kernel radius, and Average directional bands
    parameters, and is commonly used for texture classification or feature extraction.

    The GLCM tabulates how often different combinations of pixel brightness values
    (grey levels) occur in an image. It counts the number of times a pixel of value X
    lies next to a pixel of value Y, in a particular direction and distance, then
    derives statistics from this tabulation. Input is required to be integer-valued
    (cast to int internally). Applied to all bands of the input image.

    **Output:** 18 bands per input band if directional averaging is on; otherwise
    18 bands per directional pair in the kernel. Band names follow the pattern
    ``<band>_<metric>`` (e.g. ``R_contrast``, ``G_entropy``).

    **Haralick (14):**

    - **ASM** (f1): Angular Second Moment; repeated pairs
    - **CONTRAST** (f2): Local contrast
    - **CORR** (f3): Correlation between pixel pairs
    - **VAR** (f4): Variance; spread of gray-level distribution
    - **IDM** (f5): Inverse Difference Moment; homogeneity
    - **SAVG** (f6): Sum Average
    - **SVAR** (f7): Sum Variance
    - **SENT** (f8): Sum Entropy
    - **ENT** (f9): Entropy; randomness of gray-level distribution
    - **DVAR** (f10): Difference variance
    - **DENT** (f11): Difference entropy
    - **IMCORR1** (f12): Information Measure of Correlation 1
    - **IMCORR2** (f13): Information Measure of Correlation 2
    - **MAXCORR** (f14): Max Correlation Coefficient (not computed)

    **Conners (4):**

    - **DISS**: Dissimilarity
    - **INERTIA**: Inertia
    - **SHADE**: Cluster Shade
    - **PROM**: Cluster Prominence

    References: Haralick et al. (1973); Conners et al. (1984).
    """

    glcm_size = knext.IntParameter(
        "Size",
        "Size of the neighborhood to include in each GLCM (e.g. 7).",
        default_value=7,
        min_value=1,
        max_value=31,
    )

    glcm_kernel_radius = knext.IntParameter(
        "Kernel radius (for offsets)",
        "Radius in pixels for the offset kernel that defines direction/distance for GLCM. Default 1 gives 3×3 (GEE default).",
        default_value=1,
        min_value=1,
        max_value=5,
        is_advanced=True,
    )

    glcm_average = knext.BoolParameter(
        "Average directional bands",
        "If true, directional bands for each metric are averaged (default). If false, output includes separate bands per direction.",
        default_value=True,
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
            int_image = image.int()

            kernel = ee.Kernel.square(self.glcm_kernel_radius, "pixels")
            glcm_texture = int_image.glcmTexture(
                size=self.glcm_size,
                kernel=kernel,
                average=self.glcm_average,
            )

            LOGGER.warning(
                f"GLCM texture computed: size={self.glcm_size}, "
                f"kernel_radius={self.glcm_kernel_radius}, average={self.glcm_average}"
            )
            return knut.export_gee_image_connection(glcm_texture, image_connection)

        except Exception as e:
            LOGGER.error(f"GLCM texture failed: {e}")
            raise


############################################
# Spatial Statistics
############################################


@knext.node(
    name="GEE Spatial Statistics",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "SpatialStatistics.png",
    id="spatialstatistics",
    after="glcmtexture",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with one band (local spatial statistic).",
    port_type=gee_image_port_type,
)
class SpatialStatistics:
    """Computes local spatial autocorrelation: Geary's C or Moran's I.

    This node computes a local spatial statistic using Statistic, Kernel radius, and
    Input band parameters, and is commonly used to detect clustering or local variation.

    This node produces a **single-band image** where each pixel value is a local
    measure of spatial association between that pixel and its neighborhood. Both
    statistics require **one value per pixel** (one band). Use **GEE Band Reducer**
    upstream (e.g. max/mean across bands) if the input is multi-band.

    **Geary's C**

    - Formula: sum over neighbors of (x_center - x_neighbor)², divided by n².
    - **High values**: strong local variation (edges, boundaries, roads).
    - **Low values**: similar to neighbors (homogeneous areas).
    - No built-in in Earth Engine; implemented with ``neighborhoodToBands()``,
      subtract, square, sum, divide.

    **Moran's I**

    - Measures correlation between center and neighbors (deviation from local mean).
    - **Positive**: similar to neighbors (clustering).
    - **Negative**: dissimilar (dispersion).
    - **Near zero**: no clear local pattern.
    - Uses ``reduceNeighborhood(mean/variance)`` and ``neighborhoodToBands()``.

    **Kernel**

    - A square kernel with **center weight 0** and neighbor weights 1 (standard
      9×9: radius 4 → 9×9). Only non-center pixels are used in the formulas.

    **Typical use**

    - Edge/boundary detection (Geary's C).
    - Clustering vs dispersion (Moran's I).
    - Texture or context for classification.

    **Reference:** Anselin (1995), Local indicators of spatial association—LISA.
    """

    statistic = knext.StringParameter(
        "Statistic",
        "Local spatial statistic: Geary's C (local variation) or Moran's I (local correlation).",
        default_value="gearys_c",
        enum=["gearys_c", "morans_i"],
    )

    kernel_radius = knext.DoubleParameter(
        "Kernel radius",
        "Radius of the square neighborhood in pixels. Kernel size = 2×radius+1 (e.g. 4 → 9×9).",
        default_value=4.0,
        min_value=1.0,
        max_value=25.0,
    )

    input_band = knext.StringParameter(
        "Input band",
        "Band to analyze (one value per pixel). Leave empty to use the first band.",
        default_value="",
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name for the single output band. Leave empty to use default: 'gearys_c' or 'morans_i'.",
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

        try:
            image = image_connection.image

            if self.input_band:
                input_band_image = image.select(self.input_band)
            else:
                input_band_image = image.select([0])

            kernel_size = int(2 * self.kernel_radius + 1)
            row = [1] * kernel_size
            center_row = [1] * (kernel_size // 2) + [0] + [1] * (kernel_size // 2)
            rows = [
                row if i != kernel_size // 2 else center_row for i in range(kernel_size)
            ]
            kernel = ee.Kernel.fixed(
                kernel_size,
                kernel_size,
                rows,
                -kernel_size // 2,
                -kernel_size // 2,
                False,
            )

            n_neighbors = kernel_size * kernel_size - 1
            out_name = (
                self.output_band_name.strip()
                if self.output_band_name and self.output_band_name.strip()
                else self.statistic
            )

            if self.statistic == "gearys_c":
                neigh_bands = input_band_image.neighborhoodToBands(kernel)
                result = (
                    input_band_image.subtract(neigh_bands)
                    .pow(2)
                    .reduce(ee.Reducer.sum())
                    .divide(kernel_size * kernel_size)
                    .rename(out_name)
                )

            elif self.statistic == "morans_i":
                mean_img = input_band_image.reduceNeighborhood(
                    ee.Reducer.mean(), kernel
                )
                var_img = input_band_image.reduceNeighborhood(
                    ee.Reducer.variance(), kernel
                )
                centered = input_band_image.subtract(mean_img)
                neigh_bands = input_band_image.neighborhoodToBands(kernel)
                neigh_centered = neigh_bands.subtract(mean_img)
                sum_neigh_centered = neigh_centered.reduce(ee.Reducer.sum())
                denom = var_img.multiply(n_neighbors).add(1e-10)
                result = (
                    centered.multiply(sum_neigh_centered).divide(denom).rename(out_name)
                )

            else:
                raise ValueError(f"Unknown statistic: {self.statistic}")

            LOGGER.warning(
                f"Spatial statistic computed: {self.statistic}, kernel {kernel_size}x{kernel_size}"
            )
            return knut.export_gee_image_connection(result, image_connection)

        except Exception as e:
            LOGGER.error(f"Spatial statistics failed: {e}")
            raise


############################################
# Image Morphology
############################################


@knext.node(
    name="GEE Image Morphology",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "Morphology.png",
    id="imagemorphology",
    after="spatialstatistics",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with embedded image object. Input should be a single-band binary image (0 and >0 values).",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with morphology operation applied (binary 0/1 output).",
    port_type=gee_image_port_type,
)
class ImageMorphology:
    """Applies morphological operations to a binary image.

    This node applies opening or closing using Kernel radius, Kernel shape, and operation toggles,
    and is commonly used to clean up binary masks (remove noise, fill small holes).
    """

    kernel_radius = knext.IntParameter(
        "Kernel radius (pixels)",
        "Radius of the morphological kernel in pixels. A radius of 1 creates a 3x3 kernel, radius of 2 creates a 5x5 kernel, etc.",
        default_value=1,
        min_value=1,
        max_value=10,
    )

    kernel_shape = knext.StringParameter(
        "Kernel shape",
        "Shape of the morphological kernel.",
        default_value="circle",
        enum=["circle", "square"],
    )

    do_open = knext.BoolParameter(
        "Apply opening",
        "If enabled, applies opening operation (erosion followed by dilation). Removes small objects and smooths boundaries.",
        default_value=False,
    )

    do_close = knext.BoolParameter(
        "Apply closing",
        "If enabled, applies closing operation (dilation followed by erosion). Fills small holes and connects nearby objects.",
        default_value=False,
    )

    band_name = knext.StringParameter(
        "Band name",
        "Band to treat as binary mask (empty = first band). Multi-band images use one band only.",
        default_value="",
    )

    output_band_name = knext.StringParameter(
        "Output band name",
        "Name of the output morphology band (0/1).",
        default_value="morph",
    )

    use_nominal_scale = knext.BoolParameter(
        "Use NominalScale",
        "Lock the morphology output to the input image's nominal scale (native "
        "resolution). Focal operations otherwise produce a 'scale-free' image whose "
        "resolution is decided by whoever requests it, so the map View (zoom "
        "dependent) and reduceToVectors (Pixels to FC) disagree. ONLY enable this "
        "when the input has a meaningful projection (e.g. a single scene, or a "
        "composite that was reprojected with Resample/Reproject). After an IC "
        "Aggregator median/mean the native projection is lost and nominalScale "
        "degrades to ~111 km, so leave this OFF and set Scale (meters) below.",
        default_value=False,
    )

    output_scale = knext.IntParameter(
        "Scale (meters)",
        "Pixel size the morphology output is locked to. Only used when Use "
        "NominalScale is disabled. Use the native resolution of the source data "
        "(e.g. 30 for Landsat, 10 for Sentinel-2). The kernel radius (in pixels) is "
        "evaluated at this scale, and both the View and downstream vectorization use it.",
        default_value=30,
        min_value=1,
        max_value=10000,
    ).rule(knext.OneOf(use_nominal_scale, [False]), knext.Effect.SHOW)

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
        if band:
            image = image.select([band])
        else:
            names = image.bandNames().getInfo()
            if names:
                image = image.select([names[0]])

        if not self.do_open and not self.do_close:
            raise ValueError(
                "At least one morphological operation must be enabled (opening or closing)."
            )

        try:
            original_mask = image.mask()
            # Capture the input image's geometry BEFORE the focal operations.
            # focal_min/focal_max turn the result into a "scale-free",
            # *unbounded* image (geometry resets to the global extent). If we
            # don't restore the original bounds, downstream reduceToVectors
            # (Pixels to FC) tries to vectorize the whole world at the requested
            # scale, blows past maxPixels, and silently falls back to a much
            # coarser scale -> one huge, blocky polygon (while the tiled View
            # still looks correct because it only renders the current viewport).
            input_geometry = image.geometry()
            bin01 = image.gt(0).rename("bin01")

            if self.kernel_shape == "circle":
                kernel = ee.Kernel.circle(self.kernel_radius, "pixels", True)
            else:
                kernel = ee.Kernel.square(self.kernel_radius, "pixels", True)

            out = bin01

            if self.do_open:
                out = out.focal_min(self.kernel_radius, kernel=kernel).focal_max(
                    self.kernel_radius, kernel=kernel
                )
                LOGGER.warning(
                    "Applied opening: %s kernel radius=%s",
                    self.kernel_shape,
                    self.kernel_radius,
                )

            if self.do_close:
                out = out.focal_max(self.kernel_radius, kernel=kernel).focal_min(
                    self.kernel_radius, kernel=kernel
                )
                LOGGER.warning(
                    "Applied closing: %s kernel radius=%s",
                    self.kernel_shape,
                    self.kernel_radius,
                )

            morph01 = (
                out.gt(0)
                .rename(self.output_band_name)
                .toByte()
                .updateMask(original_mask)
                .clip(input_geometry)
            )

            # Lock the focal result onto a fixed grid at the chosen scale.
            # Use the CRS as a STRING (.crs()) + scale in meters: passing the
            # Projection object instead would carry its embedded transform
            # (often degraded to ~1 degree after a median/mean composite, which
            # loses the native projection) and reproject to that coarse grid.
            # With a CRS string, "scale" is honored, so both the View and
            # reduceToVectors evaluate the kernel at this exact resolution and
            # agree. When inheriting, resolve_scale returns the input's
            # nominalScale (only meaningful if the input has a real projection).
            target_crs = image.projection().crs()
            target_scale = knut.resolve_scale(
                self.use_nominal_scale, self.output_scale, image
            )
            morph01 = morph01.reproject(crs=target_crs, scale=target_scale)

            LOGGER.warning(
                "Morphology: opening=%s closing=%s output=%s",
                self.do_open,
                self.do_close,
                self.output_band_name,
            )

            return knut.export_gee_image_connection(morph01, image_connection)

        except Exception as e:
            LOGGER.error("Morphology operation failed: %s", e)
            raise
