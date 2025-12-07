"""
GEE Supervised Classification Nodes for KNIME
This module contains nodes for supervised classification using Google Earth Engine Machine Learning APIs.
Based on: https://developers.google.com/earth-engine/guides/classification
"""

import knime.extension as knext
import util.knime_utils as knut
import pandas as pd
from util.common import (
    GoogleEarthEngineConnectionObject,
    GoogleEarthEngineObjectSpec,
    google_earth_engine_port_type,
    GEEImageConnectionObject,
    GEEFeatureCollectionConnectionObject,
    GEEClassifierConnectionObject,
    gee_image_port_type,
    gee_feature_collection_port_type,
    gee_classifier_port_type,
)

# Category for GEE Supervised Classification nodes
__category = knext.category(
    path="/community/gee",
    level_id="supervised",
    name="Classification",
    description="Google Earth Engine Supervised Classification nodes",
    icon="icons/SupervisedClass.png",
    after="sampling",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/supervised/"

# Output band/property name for classification results
_ClassResult = "GEE_class"


def remap_class_values_to_continuous(training_fc, label_property):
    """Remap class values to 0-based continuous integers for GEE classifier training.

    GEE classifiers require class values to be 0-based continuous integers.
    This function maps arbitrary class values (e.g., 11, 21, 22, 90) to
    continuous integers (0, 1, 2, 3, ...).

    Returns:
        - remapped_fc: FeatureCollection with remapped class values
        - class_mapping: dict mapping original values to new values {original: new}
        - reverse_mapping: dict mapping new values to original {new: original}
    """
    import ee
    import logging

    LOGGER = logging.getLogger(__name__)

    # Get unique class values from training data
    try:
        unique_classes = (
            training_fc.aggregate_array(label_property).distinct().sort().getInfo()
        )
        unique_classes = [c for c in unique_classes if c is not None]
    except Exception as e:
        LOGGER.warning(
            f"Could not get unique classes: {e}, assuming already continuous"
        )
        return training_fc, None, None

    # Check if already 0-based continuous
    if unique_classes == list(range(len(unique_classes))):
        LOGGER.warning(
            "Class values are already 0-based continuous, no remapping needed"
        )
        return training_fc, None, None

    LOGGER.warning(f"Original class values: {unique_classes}")

    # Create mapping: original value -> 0-based index
    class_mapping = {orig: idx for idx, orig in enumerate(unique_classes)}
    reverse_mapping = {idx: orig for orig, idx in class_mapping.items()}

    LOGGER.warning(f"Class mapping: {class_mapping}")

    # Remap class values in FeatureCollection using server-side mapping
    # Build a remap list: [from1, to1, from2, to2, ...]
    remap_from = []
    remap_to = []
    for orig_val, new_val in class_mapping.items():
        remap_from.append(orig_val)
        remap_to.append(new_val)

    # Use remap() method if available, otherwise use map()
    def remap_class(feature):
        original_value = feature.get(label_property)
        # Use ee.Number to ensure proper comparison
        remapped_value = ee.Number(original_value)
        # Build chain of conditions for each class
        for idx, orig_val in enumerate(unique_classes):
            remapped_value = ee.Algorithms.If(
                ee.Number(original_value).eq(ee.Number(orig_val)),
                ee.Number(idx),
                remapped_value,
            )
        return feature.set(label_property, remapped_value)

    remapped_fc = training_fc.map(remap_class)

    return remapped_fc, class_mapping, reverse_mapping


def compute_classification_metrics(
    training_fc, classifier, label_property, reverse_mapping=None
):
    """Compute classification metrics from training data using GEE's errorMatrix.

    Optimized version: Only 2 getInfo() calls (confusion matrix + order).
    All other metrics are calculated from confusion matrix on client side.

    Returns a pandas DataFrame with per-class and overall metrics.
    """
    import ee
    import logging

    LOGGER = logging.getLogger(__name__)

    # Classify the training data
    classified_fc = training_fc.classify(classifier, _ClassResult)

    # Compute error matrix (GEE built-in, server-side)
    error_matrix = classified_fc.errorMatrix(label_property, _ClassResult)

    # Only 2 getInfo() calls: confusion matrix and order
    confusion_matrix = error_matrix.array().getInfo()
    order_raw = error_matrix.order().getInfo()

    # Process order
    num_classes = len(confusion_matrix)
    if isinstance(order_raw, list):
        unique_order = []
        seen = set()
        for val in order_raw:
            if val not in seen:
                unique_order.append(val)
                seen.add(val)
                if len(unique_order) >= num_classes:
                    break
        while len(unique_order) < num_classes:
            unique_order.append(f"Class_{len(unique_order)}")
        order = unique_order[:num_classes]
    else:
        order = [f"Class_{i}" for i in range(num_classes)]

    # Apply reverse mapping if provided
    if reverse_mapping is not None:
        mapped_order = []
        for val in order[:num_classes]:
            if isinstance(val, (int, float)) and int(val) in reverse_mapping:
                mapped_order.append(reverse_mapping[int(val)])
            else:
                mapped_order.append(val)
        order = mapped_order
        LOGGER.warning(f"Using reverse mapping to show original class values: {order}")

    LOGGER.warning(f"Computing metrics for {num_classes} classes: {order}")

    # Calculate all metrics from confusion matrix (no additional getInfo() calls)
    total_samples = sum(
        confusion_matrix[i][j] for i in range(num_classes) for j in range(num_classes)
    )
    overall_tp = sum(confusion_matrix[i][i] for i in range(num_classes))
    overall_accuracy = overall_tp / total_samples if total_samples > 0 else 0.0

    # Calculate Kappa from confusion matrix
    # Kappa = (Po - Pe) / (1 - Pe)
    # Po = overall accuracy
    # Pe = sum of (row_sum * col_sum) / total_samples^2 for each class
    pe = 0.0
    for i in range(num_classes):
        row_sum = sum(confusion_matrix[i])
        col_sum = sum(confusion_matrix[j][i] for j in range(num_classes))
        pe += (
            (row_sum * col_sum) / (total_samples * total_samples)
            if total_samples > 0
            else 0.0
        )
    kappa = (overall_accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    # Calculate per-class metrics from confusion matrix
    metrics_list = []

    for i in range(num_classes):
        class_name = order[i] if i < len(order) else f"Class_{i}"

        # Get values from confusion matrix
        tp = confusion_matrix[i][i]  # True positives (diagonal)
        row_sum = sum(confusion_matrix[i])  # Total actual samples of this class
        col_sum = sum(
            confusion_matrix[j][i] for j in range(num_classes)
        )  # Total predicted as this class

        # Calculate Precision (Consumers Accuracy) = TP / (TP + FP) = TP / col_sum
        precision = tp / col_sum if col_sum > 0 else 0.0

        # Calculate Recall (Producers Accuracy) = TP / (TP + FN) = TP / row_sum
        recall = tp / row_sum if row_sum > 0 else 0.0

        # Calculate F-measure
        f_measure = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics_list.append(
            {
                "Class": str(class_name),
                "TP": int(tp),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F-measure": round(f_measure, 4),
            }
        )

    # Add Overall row
    metrics_list.append(
        {
            "Class": "Overall",
            "TP": int(overall_tp),
            "Precision": None,
            "Recall": None,
            "F-measure": None,
            "Accuracy": round(overall_accuracy, 4),
            "Cohen's Kappa": round(kappa, 4),
        }
    )

    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    df.set_index("Class", inplace=True)

    LOGGER.warning(
        f"Created metrics table with {len(metrics_list)} rows ({num_classes} classes + 1 Overall)"
    )

    return df


def compute_validation_metrics_from_classified_fc(
    classified_fc, label_property, prediction_property=_ClassResult
):
    """Compute classification metrics from already classified FeatureCollection using GEE's errorMatrix.

    This function uses GEE's ConfusionMatrix methods to compute metrics server-side,
    minimizing data transfer between cloud and local. All accuracy metrics are computed
    on the GEE server, and only the final results are transferred.

    Args:
        classified_fc: ee.FeatureCollection that has been classified (contains both
                      label_property and prediction_property)
        label_property: Name of the property containing true class labels
        prediction_property: Name of the property containing predicted class labels
                           (default: 'GEE_class')

    Returns:
        pandas DataFrame with per-class and overall metrics.
    """
    import ee
    import logging
    import pandas as pd

    LOGGER = logging.getLogger(__name__)

    # Compute error matrix (GEE built-in, server-side)
    error_matrix = classified_fc.errorMatrix(label_property, prediction_property)

    # Use GEE's built-in methods to compute metrics server-side
    # This minimizes data transfer - only final metrics are downloaded
    # Build a server-side structure with all metrics
    metrics_dict = ee.Dictionary(
        {
            "order": error_matrix.order(),
            "confusion_array": error_matrix.array(),
            "accuracy": error_matrix.accuracy(),
            "kappa": error_matrix.kappa(),
            "producers_accuracy": error_matrix.producersAccuracy(),
            "consumers_accuracy": error_matrix.consumersAccuracy(),
        }
    )

    # Single getInfo() call to get all metrics at once
    metrics_info = metrics_dict.getInfo()

    # Extract metrics
    confusion_matrix = metrics_info["confusion_array"]
    order_raw = metrics_info["order"]
    overall_accuracy = metrics_info["accuracy"]
    kappa = metrics_info["kappa"]
    producers_accuracy_raw = metrics_info["producers_accuracy"]  # Recall (per class)
    consumers_accuracy_raw = metrics_info["consumers_accuracy"]  # Precision (per class)

    # Process producers/consumers accuracy - handle nested arrays (2D -> 1D)
    # GEE may return nested arrays where each row has one element
    if producers_accuracy_raw and isinstance(producers_accuracy_raw[0], list):
        producers_accuracy = [row[0] if row else 0.0 for row in producers_accuracy_raw]
    else:
        producers_accuracy = producers_accuracy_raw if producers_accuracy_raw else []

    if consumers_accuracy_raw and isinstance(consumers_accuracy_raw[0], list):
        consumers_accuracy = [row[0] if row else 0.0 for row in consumers_accuracy_raw]
    else:
        consumers_accuracy = consumers_accuracy_raw if consumers_accuracy_raw else []

    # Process order
    num_classes = len(confusion_matrix)
    if isinstance(order_raw, list):
        unique_order = []
        seen = set()
        for val in order_raw:
            if val not in seen:
                unique_order.append(val)
                seen.add(val)
                if len(unique_order) >= num_classes:
                    break
        while len(unique_order) < num_classes:
            unique_order.append(f"Class_{len(unique_order)}")
        order = unique_order[:num_classes]
    else:
        order = [f"Class_{i}" for i in range(num_classes)]

    LOGGER.warning(f"Computing validation metrics for {num_classes} classes: {order}")

    # Calculate per-class metrics
    # Producers Accuracy = Recall = TP / (TP + FN) = TP / row_sum
    # Consumers Accuracy = Precision = TP / (TP + FP) = TP / col_sum
    metrics_list = []

    # Get TP values from confusion matrix diagonal
    for i in range(num_classes):
        class_name = order[i] if i < len(order) else f"Class_{i}"
        tp = confusion_matrix[i][i]  # True positives (diagonal)

        # Get Recall (Producers Accuracy) and Precision (Consumers Accuracy) from GEE
        recall = float(producers_accuracy[i]) if i < len(producers_accuracy) else 0.0
        precision = float(consumers_accuracy[i]) if i < len(consumers_accuracy) else 0.0

        # Calculate F-measure
        f_measure = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics_list.append(
            {
                "Class": str(class_name),
                "TP": int(tp),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F-measure": round(f_measure, 4),
            }
        )

    # Calculate overall TP for display
    overall_tp = sum(confusion_matrix[i][i] for i in range(num_classes))

    # Add Overall row
    metrics_list.append(
        {
            "Class": "Overall",
            "TP": int(overall_tp),
            "Precision": None,
            "Recall": None,
            "F-measure": None,
            "Accuracy": round(overall_accuracy, 4),
            "Cohen's Kappa": round(kappa, 4),
        }
    )

    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    df.set_index("Class", inplace=True)

    LOGGER.warning(
        f"Created validation metrics table with {len(metrics_list)} rows ({num_classes} classes + 1 Overall)"
    )

    return df


############################################
# Label Points from Image
############################################


@knext.node(
    name="Reference Image Point Sampler",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "PointLabel.png",
    id="generatetrainingpoints",
    after="",
)
@knext.input_port(
    name="Reference Classification Image",
    description="Reference classification image (e.g., NLCD) with class values in one band.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Feature Image",
    description="Multi-band image for feature extraction (e.g., Sentinel-2, Landsat). Band values will be extracted at sample points.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with sampled training data (pixel values and class labels).",
    port_type=gee_feature_collection_port_type,
)
class LabelPointsFromImage:
    """Generates random training points from a reference classification image with band values.

    This node creates a FeatureCollection of random sample points from a reference
    classification image (e.g., NLCD) and extracts band values from a feature image
    (e.g., Sentinel-2, Landsat). The output contains both class labels and band values,
    ready for direct use with Learner nodes.

    **Use Cases:**

    - Generate training data from reference maps (NLCD, CORINE, etc.)
    - Create balanced training samples from existing classifications
    - Automate training data generation for large areas
    - Use authoritative land cover maps as training sources

    **Input Requirements:**

    - **Reference Image**: Classification image with class values (e.g., 'USGS/NLCD/NLCD2016')
    - **Feature Image**: Multi-band image for feature extraction (e.g., Sentinel-2, Landsat)
    - **Label Band**: Band name containing class values (default: first band)
    - **Bands**: Band names to extract from feature image (optional, uses all if empty)

    **Sampling Process:**

    - Generates random sample points within the intersection of both image geometries
    - Each point inherits the class value from the reference image
    - Extracts band values from the feature image at these points
    - Output FeatureCollection contains both labels and band values
    - Can be directly used with Learner nodes for training

    **Common Reference Images:**

    - **NLCD**: 'USGS/NLCD/NLCD2016' (US land cover)
    - **CORINE**: 'COPERNICUS/CORINE/V20/100m' (European land cover)
    - **ESA WorldCover**: 'ESA/WorldCover/v100' (Global land cover)
    - **Dynamic World**: 'GOOGLE/DYNAMICWORLD/V1' (Global land cover)

    **Workflow:**

    Reference Image + Feature Image → This node → Training FeatureCollection → Learner → Classifier

    **Reference:**
    Based on geemap tutorial: https://geemap.org/notebooks/32_supervised_classification/
    """

    label_band = knext.StringParameter(
        "Label band",
        """Band name containing class values in the reference image (e.g., 'landcover'). Leave empty to use first band.
        Available bands can be explored using the **GEE Image Info Extractor** node.""",
        default_value="",
    )

    bands = knext.StringParameter(
        "Bands",
        """Comma-separated list of band names to extract from feature image (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all bands. Available bands can be explored using the **GEE Image Info Extractor** node.""",
        default_value="",
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters for sampling. Should match the resolution of the reference image.",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    num_pixels = knext.IntParameter(
        "Number of pixels",
        "Number of random sample points to generate",
        default_value=5000,
        min_value=100,
        max_value=100000,
    )

    seed = knext.IntParameter(
        "Random seed",
        "Random seed for reproducible sampling",
        default_value=0,
        min_value=0,
        max_value=10000,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        reference_image_connection,
        feature_image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get images
            reference_image = reference_image_connection.image
            feature_image = feature_image_connection.image

            # Determine sampling region using intersection of both image geometries
            ref_geom = reference_image.geometry()
            feat_geom = feature_image.geometry()

            # Use intersection if both geometries are valid
            try:
                sampling_region = ref_geom.intersection(feat_geom, maxError=1)
                LOGGER.warning(
                    "Using intersection of reference and feature image geometries for sampling"
                )
            except Exception:
                # Fallback to reference image geometry
                sampling_region = ref_geom
                LOGGER.warning(
                    "Using reference image geometry for sampling (intersection failed)"
                )

            # Select label band from reference image
            if self.label_band:
                label_band = self.label_band
                reference_image_label = reference_image.select(label_band)
            else:
                # Use first band
                label_band = reference_image.bandNames().get(0).getInfo()
                reference_image_label = reference_image.select(label_band)
                LOGGER.warning(f"Using first band as label: {label_band}")

            # Select bands from feature image
            if self.bands:
                band_list = [b.strip() for b in self.bands.split(",")]
                feature_image = feature_image.select(band_list)
                LOGGER.warning(f"Selected bands from feature image: {band_list}")
            else:
                band_list = feature_image.bandNames().getInfo()
                LOGGER.warning(f"Using all bands from feature image: {band_list}")

            # Generate random sample points from reference image
            LOGGER.warning(
                f"Generating {self.num_pixels} random sample points from reference image at {self.scale}m scale"
            )

            sample_points = reference_image_label.sample(
                region=sampling_region,
                scale=self.scale,
                numPixels=self.num_pixels,
                seed=self.seed,
                geometries=True,
            )

            try:
                point_count = sample_points.size().getInfo()
                LOGGER.warning(f"Generated {point_count} sample points")
            except Exception:
                LOGGER.warning("Generated sample points (size check skipped)")

            # Extract band values from feature image at these points
            LOGGER.warning(
                f"Extracting band values from feature image at {self.scale}m scale"
            )
            sampled = feature_image.sampleRegions(
                collection=sample_points,
                properties=[label_band],  # Preserve label from reference image
                scale=self.scale,
                tileScale=1.0,
                geometries=True,  # Preserve geometry for GeoDataFrame conversion
            )

            # Get point count
            try:
                final_count = sampled.size().getInfo()
                LOGGER.warning(
                    f"Successfully generated {final_count} training points with band values and labels"
                )

                # Get class distribution
                try:
                    sample_info = sampled.limit(1000).getInfo()
                    class_values = [
                        f["properties"][label_band]
                        for f in sample_info["features"]
                        if label_band in f.get("properties", {})
                    ]
                    if class_values:
                        import collections

                        class_counts = collections.Counter(class_values)
                        LOGGER.warning(
                            f"Sample class distribution: {dict(class_counts)}"
                        )
                except Exception:
                    pass  # Ignore if we can't get distribution
            except Exception:
                LOGGER.warning("Sampling completed (size check skipped)")

            return knut.export_gee_feature_collection_connection(
                sampled, reference_image_connection
            )

        except Exception as e:
            LOGGER.error(f"Generate training points failed: {e}")
            raise


############################################
# Label Points from Feature Collection
############################################


@knext.node(
    name="Reference Feature Collection Sampler",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "LabelFeature.png",
    id="sampleregions",
    after="",
)
@knext.input_port(
    name="Training Data Connection",
    description="GEE Feature Collection connection with polygons or points containing class labels.",
    port_type=gee_feature_collection_port_type,
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with multi-band image for feature extraction.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with sampled training data (pixel values and class labels).",
    port_type=gee_feature_collection_port_type,
)
class SampleRegionsForClassification:
    """Samples pixel values from an image to create training data for classification.

    This node extracts pixel values from multi-band images to create feature vectors
    for supervised classification. It supports two sampling modes:

    **Sampling Modes:**

    - **Polygon Mode** (default): Samples all pixels within training polygons
       - Input ports: FeatureCollection with labeled polygons (first port) + Image with spectral bands (second port)
       - Use when: You have manually drawn training polygons

    - **Point Mode**: Uses pre-generated points (random/stratified/field samples) with class labels
       - Input ports: FeatureCollection of labeled point samples (first port) + Image with spectral bands (second port)
       - Use when: You already have sample points with known class labels

    **Input Requirements:**

    - **Training Data (first port)**: Feature Collection containing geometries (polygons or points) with a class label property
    - **Image (second port)**: Multi-band image with spectral information (e.g., Sentinel-2, Landsat)

    **Sampling Process:**

    - **Polygon Mode**: Extracts all pixels within each training polygon
    - **Point Mode**: Uses provided point samples and extracts band values at each point
    - Creates a FeatureCollection with one feature per pixel (band values + class label + geometry)
    - Supports large datasets with efficient server-side processing
    - Output can be directly used for training classifiers or converted to table if needed

    **Common Use Cases:**

    - Create training data for land cover classification
    - Sample vegetation indices for crop type mapping
    - Extract spectral signatures for material identification
    - Generate balanced datasets for machine learning
    - Use nodes such as "Label Random Points from Reference" to derive labeled samples from reference maps

    **Performance Tips:**

    - Lower scale values provide more samples but may be slower
    - Use tile_scale > 1.0 for large training areas
    - Ensure sufficient samples per class (minimum 100-200 recommended)
    - Point mode: Provide a sufficient number of labeled point samples for each class

    **Workflow:**

    1. This node → Output: Training FeatureCollection
    2. Training FeatureCollection → Train Classifier nodes → Output: Classifier Connection
    3. Image + Classifier Connection → Classify Image → Output: Classified Image

    **Note:** The output FeatureCollection can be directly used for training. If you need a table format,
    use "Feature Collection to Table" node after this node.
    """

    sampling_mode = knext.StringParameter(
        "Sampling mode",
        """Method for generating training samples. 
        Available modes:
        
        - **Polygon Mode** (default): Samples all pixels within training polygons. 
        Input ports: FeatureCollection with labeled polygons (first port) + Image with spectral bands (second port). 
        Use when: You have manually drawn training polygons

        - **Point Mode**: Uses pre-generated points (random/stratified/field samples) with class labels. 
        Input ports: FeatureCollection of labeled point samples (first port) + Image with spectral bands (second port). 
        Use when: You already have sample points with known class labels
        """,
        default_value="Polygon",
        enum=["Polygon", "Point"],
    )

    label_property = knext.StringParameter(
        "Label property",
        "Property name containing class labels in the training Feature Collection (e.g., 'class', 'LC', 'landcover').",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands",
        """Comma-separated list of band names to use for training (e.g., 'B2,B3,B4,B8' for Sentinel-2). 
        Leave empty to use all bands. Available bands can be explored using the **GEE Image Info Extractor** node.""",
        default_value="",
    )

    scale = knext.IntParameter(
        "Scale (meters)",
        "The scale in meters for sampling. Lower values provide higher resolution but may be slower.",
        default_value=30,
        min_value=1,
        max_value=10000,
    )

    tile_scale = knext.DoubleParameter(
        "Tile scale",
        "Tile scale for performance optimization (1.0 = default, higher = faster for large areas)",
        default_value=1.0,
        min_value=0.1,
        max_value=16.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
        image_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training feature collection and image from connections
            training_fc = fc_connection.feature_collection
            image = image_connection.image

            # Select bands if specified
            if self.bands:
                band_list = [b.strip() for b in self.bands.split(",")]
                image = image.select(band_list)
                LOGGER.warning(f"Selected bands: {band_list}")
            else:
                # Use all bands
                band_list = image.bandNames().getInfo()
                LOGGER.warning(f"Using all bands: {band_list}")

            # Validate training data type
            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError(
                    "Training data input must be a FeatureCollection. "
                    "Please provide labeled polygons or points as a FeatureCollection."
                )

            # Sample based on mode
            if self.sampling_mode == "Polygon":
                # Polygon mode: sample all pixels within polygons
                LOGGER.warning(
                    f"Polygon mode: Sampling image at {self.scale}m scale with label property: {self.label_property}"
                )
                sampled = image.sampleRegions(
                    collection=training_fc,
                    properties=[self.label_property],
                    scale=self.scale,
                    tileScale=self.tile_scale,
                    geometries=True,  # Preserve geometry for GeoDataFrame conversion
                )

                # Get sample count for logging
                try:
                    sample_count = sampled.size().getInfo()
                    LOGGER.warning(
                        f"Successfully sampled {sample_count} training pixels from polygons"
                    )
                except Exception:
                    LOGGER.warning("Sampling completed (size check skipped)")

                # Return FeatureCollection directly
                return knut.export_gee_feature_collection_connection(
                    sampled, image_connection
                )

            else:
                # Point mode: sample at provided point locations
                LOGGER.warning(
                    "Point mode: Using supplied FeatureCollection of training points"
                )

                sample_points = training_fc

                sample_feature_info = sample_points.first().getInfo()
                if sample_feature_info is None:
                    raise ValueError(
                        "Training points FeatureCollection is empty. "
                        "Provide at least one feature with a class label."
                    )

                available_props = list(sample_feature_info.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                band_names = set(band_list) if band_list else set()

                preferred_label = (self.label_property or "").strip()
                label_prop = None
                if preferred_label and preferred_label in available_props:
                    label_prop = preferred_label
                    LOGGER.warning(
                        f"Using user-specified label property '{label_prop}'"
                    )
                else:
                    for prop in available_props:
                        if prop not in system_props and prop not in band_names:
                            label_prop = prop
                            LOGGER.warning(
                                f"Auto-detected label property '{label_prop}' in training points"
                            )
                            break

                if label_prop is None:
                    raise ValueError(
                        "Could not determine the label property in training points. "
                        "Please ensure the FeatureCollection has a class label column and/or set the 'Label Property' parameter."
                    )

                # Extract values from the image at the supplied point locations
                sampled = image.sampleRegions(
                    collection=sample_points,
                    properties=[label_prop],
                    scale=self.scale,
                    tileScale=self.tile_scale,
                    geometries=True,  # Preserve geometry for GeoDataFrame conversion
                )

                try:
                    sample_count = sampled.size().getInfo()
                    LOGGER.warning(
                        f"Successfully sampled {sample_count} training pixels from provided points"
                    )
                except Exception:
                    LOGGER.warning("Sampling completed (size check skipped)")

                return knut.export_gee_feature_collection_connection(
                    sampled, image_connection
                )

        except Exception as e:
            LOGGER.error(f"Sample regions for classification failed: {e}")
            raise


############################################
# Training Data Partitioner
############################################


@knext.node(
    name="Feature Collection Partitioner",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCpartitioner.png",
    id="trainingdatapartitioner",
    after="sampleregions",
)
@knext.input_port(
    name="Training Data",
    description="GEE Feature Collection connection with training data to split into train and validation sets.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="Training Set",
    description="GEE Feature Collection connection with training subset.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="Validation Set",
    description="GEE Feature Collection connection with validation subset.",
    port_type=gee_feature_collection_port_type,
)
class TrainingDataPartitioner:
    """Splits training data into training and validation sets.

    This node randomly partitions a FeatureCollection into two subsets:
    - **Training Set**: Used for training the classifier
    - **Validation Set**: Used for evaluating model performance (unseen data)

    **Use Cases:**

    - Create independent validation set for unbiased model evaluation
    - Split data before training to assess generalization performance
    - Enable proper model validation workflow

    **Workflow:**

    1. Training Data → This node → Training Set + Validation Set
    2. Training Set → Learner nodes → Trained Classifier
    3. Validation Set + Trained Classifier → Feature Collection Predictor → Predictions
    4. Validation Set + Predictions → Classifier Scorer → Validation Metrics

    **Split Method:**

    - Uses random sampling with specified split ratio
    - Ensures reproducible results with random seed
    - Removes temporary 'random' column after splitting

    **Parameters:**

    - **Train Fraction**: Fraction of data for training (default: 0.8 = 80% training, 20% validation)
    - **Random Seed**: For reproducible splits
    """

    train_fraction = knext.DoubleParameter(
        "Train fraction",
        "Fraction of data to use for training (0.0-1.0). Remaining data will be used for validation.",
        default_value=0.8,
        min_value=0.1,
        max_value=0.99,
    )

    seed = knext.IntParameter(
        "Random seed",
        "Random seed for reproducible data splitting",
        default_value=0,
        min_value=0,
        max_value=10000,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_data_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training FeatureCollection
            training_fc = training_data_connection.feature_collection

            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError("Input must be a FeatureCollection with training data")

            # Get data size for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(
                    f"Partitioning {fc_size} samples into training and validation sets"
                )
            except Exception:
                LOGGER.warning(
                    "Partitioning FeatureCollection into training and validation sets"
                )

            # Add random column for splitting
            sample_with_random = training_fc.randomColumn(seed=self.seed)

            # Split into training and validation sets
            training_fc_split = sample_with_random.filter(
                ee.Filter.lte("random", self.train_fraction)
            )
            validation_fc_split = sample_with_random.filter(
                ee.Filter.gt("random", self.train_fraction)
            )

            # Remove the 'random' column from both sets
            # Get all property names except 'random' by getting property names from original FC
            first_feature = training_fc.first()
            all_properties = first_feature.propertyNames().getInfo()
            properties_to_keep = [p for p in all_properties if p != "random"]

            # Select only the original properties (exclude 'random')
            training_fc_clean = training_fc_split.select(properties_to_keep)
            validation_fc_clean = validation_fc_split.select(properties_to_keep)

            # Log split sizes
            try:
                train_size = training_fc_clean.size().getInfo()
                val_size = validation_fc_clean.size().getInfo()
                LOGGER.warning(
                    f"Split complete: {train_size} training samples ({self.train_fraction*100:.1f}%), "
                    f"{val_size} validation samples ({(1-self.train_fraction)*100:.1f}%)"
                )
            except Exception:
                LOGGER.warning("Split complete (size check skipped)")

            # Export both FeatureCollections
            training_connection = knut.export_gee_feature_collection_connection(
                training_fc_clean, training_data_connection
            )
            validation_connection = knut.export_gee_feature_collection_connection(
                validation_fc_clean, training_data_connection
            )

            return training_connection, validation_connection

        except Exception as e:
            LOGGER.error(f"Training data partitioning failed: {e}")
            raise


############################################
# Train Random Forest Classifier
############################################


@knext.node(
    name="GEE Random Forest Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "RandomForest.png",
    id="randomforestlearner",
    after="sampleregions",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained Random Forest model.",
    port_type=gee_classifier_port_type,
)
class RandomForestLearner:
    """Trains a Random Forest classifier using training data.

    Random Forest is a robust ensemble method that combines multiple decision trees
    to create a strong classifier. It is one of the most popular and effective
    algorithms for remote sensing classification.

    **Algorithm Details:**

    - **Ensemble Method**: Combines multiple decision trees via voting
    - **Robust**: Handles noise and outliers well
    - **Feature Importance**: Automatically calculates feature importance
    - **No Overfitting**: Built-in regularization through ensemble

    **Parameters:**

    - **Number of Trees**: More trees = better accuracy but slower training (default: 100)
    - **Variables per Split**: Number of features considered at each split (default: sqrt of total)
    - **Min Leaf Population**: Minimum samples required in a leaf node (default: 1)
    - **Bag Fraction**: Fraction of samples used for each tree (default: 0.5)
    - **Max Nodes**: Maximum nodes per tree to limit complexity (default: None = unlimited)

    **Common Use Cases:**

    - Land cover classification
    - Crop type mapping
    - Urban area detection
    - Vegetation type identification

    **Performance:**

    - Training dataset limited to ~100MB (typically < 200,000 samples with 100 bands)
    - Generally effective for most remote sensing applications
    - Can handle large numbers of features/bands

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    label_property = knext.StringParameter(
        "Label property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all properties except label. Available bands/features can be explored using the 
        **GEE Image Info Extractor** node.""",
        default_value="",
    )

    number_of_trees = knext.IntParameter(
        "Number of trees",
        "Number of decision trees in the random forest ensemble",
        default_value=100,
        min_value=10,
        max_value=1000,
    )

    variables_per_split = knext.IntParameter(
        "Variables per split",
        "Number of variables to consider at each split (default: sqrt of total features). Set to 0 for auto.",
        default_value=0,
        min_value=0,
        max_value=100,
        is_advanced=True,
    )

    min_leaf_population = knext.IntParameter(
        "Min leaf population",
        "Minimum number of samples required in a leaf node",
        default_value=1,
        min_value=1,
        max_value=1000,
        is_advanced=True,
    )

    bag_fraction = knext.DoubleParameter(
        "Bag fraction",
        "Fraction of samples to use for training each tree (0.0-1.0)",
        default_value=0.5,
        min_value=0.1,
        max_value=1.0,
        is_advanced=True,
    )

    max_nodes = knext.IntParameter(
        "Max nodes",
        "Maximum number of nodes per tree (0 = unlimited). Use to limit model size.",
        default_value=0,
        min_value=0,
        max_value=1000000,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_data_connection,
    ):
        import ee
        import logging
        import math

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training FeatureCollection
            training_fc = training_data_connection.feature_collection

            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection from 'Sample Regions for Classification' or 'Generate Training Points from Reference Image'"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(
                    f"Training Random Forest with {fc_size} samples (from FeatureCollection)"
                )
            except Exception:
                LOGGER.warning(
                    "Training Random Forest with FeatureCollection (size check skipped)"
                )

            # Get available properties from first feature
            try:
                sample_feature = training_fc.first().getInfo()
                available_properties = list(sample_feature.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                available_features = [
                    p for p in available_properties if p not in system_props
                ]
                LOGGER.warning(
                    f"Available properties: {available_features} (excluding system properties)"
                )
            except Exception as e:
                LOGGER.warning(f"Could not inspect FeatureCollection properties: {e}")
                available_features = []

            # Determine feature list (bands to use)
            if self.bands:
                # Use specified bands
                feature_list = [b.strip() for b in self.bands.split(",")]
                # Validate that specified bands exist
                missing_bands = [b for b in feature_list if b not in available_features]
                if missing_bands:
                    LOGGER.warning(
                        f"Warning: Some specified bands not found in FeatureCollection: {missing_bands}"
                    )
                feature_list = [b for b in feature_list if b in available_features]
            else:
                # Use all properties except label and system properties
                feature_list = [
                    p for p in available_features if p != self.label_property
                ]

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Label property: {self.label_property}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for training: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build classifier parameters
            classifier_params = {
                "numberOfTrees": self.number_of_trees,
                "minLeafPopulation": self.min_leaf_population,
                "bagFraction": self.bag_fraction,
            }

            # Set variables per split (auto if 0)
            if self.variables_per_split > 0:
                classifier_params["variablesPerSplit"] = self.variables_per_split
            else:
                # Auto: sqrt of number of features
                if feature_list:
                    classifier_params["variablesPerSplit"] = int(
                        math.sqrt(len(feature_list))
                    )
                # If feature_list is None, GEE will auto-determine

            # Set max nodes if specified
            if self.max_nodes > 0:
                classifier_params["maxNodes"] = self.max_nodes

            # Remap class values to 0-based continuous integers
            # GEE classifiers require continuous integer class values starting from 0
            try:
                remapped_fc, class_mapping, reverse_mapping = (
                    remap_class_values_to_continuous(training_fc, self.label_property)
                )
                if class_mapping is not None:
                    LOGGER.warning(
                        f"Remapped {len(class_mapping)} class values to 0-based continuous integers"
                    )
                    training_fc_for_training = remapped_fc
                else:
                    training_fc_for_training = training_fc
                    reverse_mapping = None
            except Exception as e:
                LOGGER.warning(
                    f"Could not remap class values (may already be continuous): {e}. Using original values."
                )
                training_fc_for_training = training_fc
                reverse_mapping = None

            # Create and train classifier
            classifier = ee.Classifier.smileRandomForest(**classifier_params)

            # Train classifier
            trained_classifier = classifier.train(
                features=training_fc_for_training,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained Random Forest classifier with {self.number_of_trees} trees"
            )

            # Create classifier connection object using credentials from training data connection
            classifier_connection = knut.export_gee_classifier_connection(
                classifier=trained_classifier,
                existing_connection=training_data_connection,
                training_data=training_fc_for_training,  # Store remapped training data
                label_property=self.label_property,
                reverse_mapping=reverse_mapping,  # Store reverse mapping for prediction
                input_properties=feature_list,  # Store bands/features used for training
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"Random Forest classifier training failed: {e}")
            raise


############################################
# CART Learner
############################################


@knext.node(
    name="GEE CART Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "cart.png",
    id="cartlearner",
    after="randomforestlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained CART model.",
    port_type=gee_classifier_port_type,
)
class CARTLearner:
    """Trains a CART (Classification and Regression Tree) classifier.

    CART is a simple, interpretable decision tree algorithm that is fast and
    effective for many classification tasks. It's particularly useful when
    you need to understand which features are most important for classification.

    **Algorithm Details:**

    - **Decision Tree**: Single tree structure (easy to interpret)
    - **Fast Training**: Quick to train compared to ensemble methods
    - **Interpretable**: Can visualize the decision tree structure
    - **Feature Selection**: Automatically selects important features

    **Parameters:**

    - **Max Nodes**: Maximum number of nodes in the tree (default: 10000)
    - **Min Leaf Population**: Minimum samples required in a leaf node (default: 1)

    **Note:** CART does not support `variablesPerSplit` parameter (only RandomForest does).

    **Common Use Cases:**

    - Quick classification experiments
    - When interpretability is important
    - Baseline model for comparison
    - Small to medium-sized datasets

    **Performance:**

    - Faster training than Random Forest
    - May overfit on complex datasets
    - Generally less accurate than ensemble methods

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    label_property = knext.StringParameter(
        "Label property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8').
        Leave empty to use all properties except label. Available bands/features can be explored using the 
        **GEE Image Info Extractor** node.""",
        default_value="",
    )

    max_nodes = knext.IntParameter(
        "Max nodes",
        "Maximum number of nodes in the decision tree",
        default_value=10000,
        min_value=100,
        max_value=1000000,
    )

    min_leaf_population = knext.IntParameter(
        "Min leaf population",
        "Minimum number of samples required in a leaf node",
        default_value=1,
        min_value=1,
        max_value=1000,
        is_advanced=True,
    )

    # Note: CART does not support variablesPerSplit parameter (only RandomForest does)
    # This parameter is kept for UI consistency but not used in classifier creation

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_data_connection,
    ):
        import ee
        import logging
        import math

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training FeatureCollection
            training_fc = training_data_connection.feature_collection

            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection from 'Sample Regions for Classification' or 'Generate Training Points from Reference Image'"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(
                    f"Training CART classifier with {fc_size} samples (from FeatureCollection)"
                )
            except Exception:
                LOGGER.warning(
                    "Training CART classifier with FeatureCollection (size check skipped)"
                )

            # Get available properties from first feature
            try:
                sample_feature = training_fc.first().getInfo()
                available_properties = list(sample_feature.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                available_features = [
                    p for p in available_properties if p not in system_props
                ]
                LOGGER.warning(
                    f"Available properties: {available_features} (excluding system properties)"
                )
            except Exception as e:
                LOGGER.warning(f"Could not inspect FeatureCollection properties: {e}")
                available_features = []

            # Determine feature list (bands to use)
            if self.bands:
                # Use specified bands
                feature_list = [b.strip() for b in self.bands.split(",")]
                # Validate that specified bands exist
                missing_bands = [b for b in feature_list if b not in available_features]
                if missing_bands:
                    LOGGER.warning(
                        f"Warning: Some specified bands not found in FeatureCollection: {missing_bands}"
                    )
                feature_list = [b for b in feature_list if b in available_features]
            else:
                # Use all properties except label and system properties
                feature_list = [
                    p for p in available_features if p != self.label_property
                ]

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Label property: {self.label_property}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for training: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build classifier parameters
            # Note: CART only supports maxNodes and minLeafPopulation (variablesPerSplit is not supported)
            classifier_params = {
                "maxNodes": self.max_nodes,
                "minLeafPopulation": self.min_leaf_population,
            }

            # Remap class values to 0-based continuous integers
            # GEE classifiers require continuous integer class values starting from 0
            try:
                remapped_fc, class_mapping, reverse_mapping = (
                    remap_class_values_to_continuous(training_fc, self.label_property)
                )
                if class_mapping is not None:
                    LOGGER.warning(
                        f"Remapped {len(class_mapping)} class values to 0-based continuous integers"
                    )
                    training_fc_for_training = remapped_fc
                else:
                    training_fc_for_training = training_fc
                    reverse_mapping = None
            except Exception as e:
                LOGGER.warning(
                    f"Could not remap class values (may already be continuous): {e}. Using original values."
                )
                training_fc_for_training = training_fc
                reverse_mapping = None

            # Create and train classifier
            classifier = ee.Classifier.smileCart(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc_for_training,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained CART classifier with max {self.max_nodes} nodes"
            )

            # Create classifier connection object using credentials from training data connection
            classifier_connection = knut.export_gee_classifier_connection(
                classifier=trained_classifier,
                existing_connection=training_data_connection,
                training_data=training_fc_for_training,  # Store remapped training data
                label_property=self.label_property,
                reverse_mapping=reverse_mapping,  # Store reverse mapping for prediction
                input_properties=feature_list,  # Store bands/features used for training
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"CART classifier training failed: {e}")
            raise


############################################
# SVM Learner
############################################


@knext.node(
    name="GEE SVM Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "SVM.png",
    id="svmlearner",
    after="cartlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained SVM model.",
    port_type=gee_classifier_port_type,
)
class SVMLearner:
    """Trains a Support Vector Machine (SVM) classifier.

    SVM is a powerful algorithm that works well for high-dimensional data and
    can handle non-linear classification through kernel functions.

    **Algorithm Details:**

    - **Kernel Methods**: Supports linear, RBF, and polynomial kernels
    - **High-Dimensional**: Effective for many features/bands
    - **Margin Maximization**: Finds optimal decision boundary
    - **Binary Classification**: Naturally handles binary classification

    **Parameters:**

    - **Kernel Type**: Type of kernel function (RBF, Linear, Polynomial)
    - **Gamma**: RBF kernel parameter (default: 0.001)
    - **Cost**: Regularization parameter (default: 1.0)
    - **Degree**: Polynomial kernel degree (default: 3)

    **Common Use Cases:**

    - High-dimensional feature spaces
    - Binary classification tasks
    - When you need a different approach than tree-based methods

    **Performance:**

    - Can be slower for large datasets
    - Effective for high-dimensional data
    - Requires careful parameter tuning

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    label_property = knext.StringParameter(
        "Label property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all properties except label. Available bands/features can be explored using the 
        **GEE Image Info Extractor** node.""",
        default_value="",
    )

    kernel_type = knext.StringParameter(
        "Kernel type",
        "Type of kernel function for SVM",
        default_value="RBF",
        enum=["RBF", "Linear", "Polynomial"],
    )

    gamma = knext.DoubleParameter(
        "Gamma",
        "RBF kernel parameter (only used for RBF kernel)",
        default_value=0.001,
        min_value=0.0001,
        max_value=10.0,
        is_advanced=True,
    ).rule(knext.OneOf(kernel_type, ["RBF"]), knext.Effect.SHOW)

    cost = knext.DoubleParameter(
        "Cost (C)",
        "Regularization parameter (higher = more complex model)",
        default_value=1.0,
        min_value=0.01,
        max_value=100.0,
        is_advanced=True,
    )

    degree = knext.IntParameter(
        "Degree",
        "Polynomial kernel degree (only used for Polynomial kernel)",
        default_value=3,
        min_value=2,
        max_value=10,
        is_advanced=True,
    ).rule(knext.OneOf(kernel_type, ["Polynomial"]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_data_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training FeatureCollection
            training_fc = training_data_connection.feature_collection

            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection from 'Sample Regions for Classification' or 'Generate Training Points from Reference Image'"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(
                    f"Training SVM classifier with {fc_size} samples (from FeatureCollection)"
                )
            except Exception:
                LOGGER.warning(
                    "Training SVM classifier with FeatureCollection (size check skipped)"
                )

            # Get available properties from first feature
            try:
                sample_feature = training_fc.first().getInfo()
                available_properties = list(sample_feature.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                available_features = [
                    p for p in available_properties if p not in system_props
                ]
                LOGGER.warning(
                    f"Available properties: {available_features} (excluding system properties)"
                )
            except Exception as e:
                LOGGER.warning(f"Could not inspect FeatureCollection properties: {e}")
                available_features = []

            # Determine feature list (bands to use)
            if self.bands:
                # Use specified bands
                feature_list = [b.strip() for b in self.bands.split(",")]
                # Validate that specified bands exist
                missing_bands = [b for b in feature_list if b not in available_features]
                if missing_bands:
                    LOGGER.warning(
                        f"Warning: Some specified bands not found in FeatureCollection: {missing_bands}"
                    )
                feature_list = [b for b in feature_list if b in available_features]
            else:
                # Use all properties except label and system properties
                feature_list = [
                    p for p in available_features if p != self.label_property
                ]

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Label property: {self.label_property}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for training: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build classifier parameters
            # GEE libsvm parameters: kernelType, cost, gamma (for RBF), degree (for Polynomial)
            classifier_params = {"cost": self.cost}

            # Add kernel-specific parameters
            if self.kernel_type == "RBF":
                classifier_params["kernelType"] = "RBF"
                classifier_params["gamma"] = self.gamma
            elif self.kernel_type == "Linear":
                classifier_params["kernelType"] = "Linear"
            elif self.kernel_type == "Polynomial":
                classifier_params["kernelType"] = "Polynomial"
                classifier_params["degree"] = self.degree

            # Remap class values to 0-based continuous integers
            # GEE classifiers require continuous integer class values starting from 0
            try:
                remapped_fc, class_mapping, reverse_mapping = (
                    remap_class_values_to_continuous(training_fc, self.label_property)
                )
                if class_mapping is not None:
                    LOGGER.warning(
                        f"Remapped {len(class_mapping)} class values to 0-based continuous integers"
                    )
                    training_fc_for_training = remapped_fc
                else:
                    training_fc_for_training = training_fc
                    reverse_mapping = None
            except Exception as e:
                LOGGER.warning(
                    f"Could not remap class values (may already be continuous): {e}. Using original values."
                )
                training_fc_for_training = training_fc
                reverse_mapping = None

            # Create and train classifier
            # Note: GEE uses libsvm for SVM
            classifier = ee.Classifier.libsvm(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc_for_training,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained SVM classifier with {self.kernel_type} kernel"
            )

            # Create classifier connection object using credentials from training data connection
            classifier_connection = knut.export_gee_classifier_connection(
                classifier=trained_classifier,
                existing_connection=training_data_connection,
                training_data=training_fc_for_training,  # Store remapped training data
                label_property=self.label_property,
                reverse_mapping=reverse_mapping,  # Store reverse mapping for prediction
                input_properties=feature_list,  # Store bands/features used for training
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"SVM classifier training failed: {e}")
            raise


############################################
# Naive Bayes Learner
############################################


@knext.node(
    name="GEE Naive Bayes Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "NaiveBayes.png",
    id="naivebayeslearner",
    after="svmlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained Naive Bayes model.",
    port_type=gee_classifier_port_type,
)
class NaiveBayesLearner:
    """Trains a Naive Bayes classifier using SMILE implementation.

    Naive Bayes is a simple probabilistic classifier based on Bayes' theorem.
    It's fast, requires minimal training data, and works well for many applications.

    **⚠️ Important Feature Requirements:**

    - **Non-negative Integer Features**: This implementation requires features to be
      non-negative integers (positive integer feature vectors). Negative values will
      be discarded automatically.
    - **Continuous Features**: If your features are continuous (e.g., reflectance values),
      you may need to discretize them first (e.g., using quantiles and converting to integers).
    - **Feature Preprocessing**: Consider using `.toInt()` or remapping continuous values
      to integer ranges before training.

    **Algorithm Details:**

    - **Probabilistic**: Uses probability distributions for classification
    - **Fast**: Very quick training and prediction
    - **Simple**: Minimal assumptions about data distribution
    - **Naive Assumption**: Assumes feature independence (often violated but still effective)
    - **Smoothing**: Uses Laplace smoothing (lambda parameter) to handle zero probabilities

    **Parameters:**

    - **Lambda (Smoothing)**: Smoothing parameter for Laplace smoothing (default: 0.0).
      Higher values add more smoothing, which can help with sparse data.

    **Common Use Cases:**

    - Quick classification experiments
    - When you have limited training data
    - Baseline model for comparison
    - Real-time classification applications
    - Discrete/categorical feature spaces

    **Performance:**

    - Fastest training among all classifiers
    - Generally less accurate than Random Forest or SVM
    - Good for preliminary analysis
    - Works best with discrete/categorical features

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    label_property = knext.StringParameter(
        "Label property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all properties except label. ⚠️ Note: Features must be non-negative integers. 
        Available bands/features can be explored using the **GEE Image Info Extractor** node.""",
        default_value="",
    )

    lambda_smoothing = knext.DoubleParameter(
        "Lambda (Smoothing)",
        "Smoothing parameter for Laplace smoothing (default: 0.0). Higher values add more smoothing.",
        default_value=0.0,
        min_value=0.0,
        max_value=10.0,
        is_advanced=True,
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_data_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get training FeatureCollection
            training_fc = training_data_connection.feature_collection

            if not isinstance(training_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection from 'Sample Regions for Classification' or 'Generate Training Points from Reference Image'"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(
                    f"Training Naive Bayes classifier with {fc_size} samples (from FeatureCollection)"
                )
            except Exception:
                LOGGER.warning(
                    "Training Naive Bayes classifier with FeatureCollection (size check skipped)"
                )

            # Get available properties from first feature
            try:
                sample_feature = training_fc.first().getInfo()
                available_properties = list(sample_feature.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                available_features = [
                    p for p in available_properties if p not in system_props
                ]
                LOGGER.warning(
                    f"Available properties: {available_features} (excluding system properties)"
                )
            except Exception as e:
                LOGGER.warning(f"Could not inspect FeatureCollection properties: {e}")
                available_features = []

            # Determine feature list (bands to use)
            if self.bands:
                # Use specified bands
                feature_list = [b.strip() for b in self.bands.split(",")]
                # Validate that specified bands exist
                missing_bands = [b for b in feature_list if b not in available_features]
                if missing_bands:
                    LOGGER.warning(
                        f"Warning: Some specified bands not found in FeatureCollection: {missing_bands}"
                    )
                feature_list = [b for b in feature_list if b in available_features]
            else:
                # Use all properties except label and system properties
                feature_list = [
                    p for p in available_features if p != self.label_property
                ]

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Label property: {self.label_property}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for training: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Warn about feature requirements
            LOGGER.warning(
                "⚠️ Naive Bayes requires non-negative integer features. "
                "Negative values will be discarded. Continuous features may need discretization."
            )

            # Remap class values to 0-based continuous integers
            # GEE classifiers require continuous integer class values starting from 0
            try:
                remapped_fc, class_mapping, reverse_mapping = (
                    remap_class_values_to_continuous(training_fc, self.label_property)
                )
                if class_mapping is not None:
                    LOGGER.warning(
                        f"Remapped {len(class_mapping)} class values to 0-based continuous integers"
                    )
                    training_fc_for_training = remapped_fc
                else:
                    training_fc_for_training = training_fc
                    reverse_mapping = None
            except Exception as e:
                LOGGER.warning(
                    f"Could not remap class values (may already be continuous): {e}. Using original values."
                )
                training_fc_for_training = training_fc
                reverse_mapping = None

            # Create and train classifier
            # Note: smileNaiveBayes supports optional lambda parameter for smoothing
            classifier_params = {}
            if self.lambda_smoothing > 0.0:
                classifier_params["lambda"] = self.lambda_smoothing

            classifier = ee.Classifier.smileNaiveBayes(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc_for_training,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained Naive Bayes classifier (lambda={self.lambda_smoothing})"
            )

            # Create classifier connection object using credentials from training data connection
            classifier_connection = knut.export_gee_classifier_connection(
                classifier=trained_classifier,
                existing_connection=training_data_connection,
                training_data=training_fc_for_training,  # Store remapped training data
                label_property=self.label_property,
                reverse_mapping=reverse_mapping,  # Store reverse mapping for prediction
                input_properties=feature_list,  # Store bands/features used for training
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"Naive Bayes classifier training failed: {e}")
            raise


############################################
# Image Class Predictor
############################################


@knext.node(
    name="Image Class Predictor",
    node_type=knext.NodeType.PREDICTOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImagePredictor.png",
    id="imageclasspredictor",
    after="naivebayeslearner",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with image to classify.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained classifier model (from Learner nodes).",
    port_type=gee_classifier_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with classified image.",
    port_type=gee_image_port_type,
)
class ImageClassPredictor:
    """Applies a trained classifier to classify an image.

    This node uses a trained machine learning model to classify pixels in an image,
    producing a classification map with predicted class labels and probabilities.

    **Input Requirements:**

    - **Image**: Multi-band image (must contain the bands used during training)
    - **Classifier**: Trained classifier from Learner nodes (Random Forest Learner, CART Learner, SVM Learner, or Naive Bayes Learner)

    **Automatic Band Selection:**

    - The node automatically uses the same bands that were used during training
    - Bands are inherited from the classifier connection (no manual selection needed)
    - The input image must contain all the bands used during training

    **Output:**

    The node outputs an image with one band:
    - **GEE_class**: Class IDs (integer values, remapped to original class values if applicable)

    **Common Use Cases:**

    - Land cover mapping
    - Crop type classification
    - Urban area detection
    - Vegetation type mapping
    - Change detection analysis

    **Post-Processing:**

    - Use classification statistics nodes to analyze results
    - Apply smoothing filters if needed
    - Convert to Feature Collection for further analysis

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        classifier_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and classifier from connections
            image = image_connection.image
            classifier = classifier_connection.classifier

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please use a trained classifier from Learner nodes (Random Forest Learner, CART Learner, SVM Learner, or Naive Bayes Learner)."
                )

            # Get input properties (bands) from classifier connection
            input_properties = classifier_connection.input_properties
            if input_properties:
                # Select only the bands used during training
                image = image.select(input_properties)
                LOGGER.warning(
                    f"Using training bands for classification: {input_properties}"
                )
            else:
                LOGGER.warning(
                    "No input properties found in classifier connection, using all bands"
                )

            # Classify the image - get prediction (class IDs)
            # Output band name is set to _ClassResult ('GEE_class')
            classified_image = image.classify(classifier, outputName=_ClassResult)

            # Map prediction results back to original class values if reverse_mapping exists
            reverse_mapping = classifier_connection.reverse_mapping
            if reverse_mapping is not None:
                LOGGER.warning(
                    f"Mapping predictions back to original class values using reverse mapping: {reverse_mapping}"
                )

                # Build remap lists for Image.remap()
                # reverse_mapping format: {0: 11, 1: 21, 2: 22, ...}
                remap_from = list(reverse_mapping.keys())  # [0, 1, 2, ...]
                remap_to = [
                    reverse_mapping[idx] for idx in remap_from
                ]  # [11, 21, 22, ...]

                # Remap the classification band (using _ClassResult 'GEE_class')
                remapped_band = classified_image.select(_ClassResult).remap(
                    remap_from, remap_to
                )

                # Replace the classification band with remapped values
                classified_image = classified_image.addBands(
                    remapped_band, overwrite=True
                )

                LOGGER.warning(
                    "Successfully remapped predictions to original class values"
                )

            LOGGER.warning("Successfully classified image")

            return knut.export_gee_image_connection(classified_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Image classification failed: {e}")
            raise


############################################
# Feature Collection Predictor
############################################


@knext.node(
    name="Feature Collection Predictor",
    node_type=knext.NodeType.PREDICTOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCpredictor.png",
    id="featurecollectionpredictor",
    after="imageclasspredictor",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with features to classify (must contain the same band/property names used for training).",
    port_type=gee_feature_collection_port_type,
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained classifier model (from Learner nodes).",
    port_type=gee_classifier_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with prediction results added as a new property.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionPredictor:
    """Applies a trained classifier to classify features in a Feature Collection.

    This node uses a trained machine learning model to predict class labels
    for each feature in a Feature Collection. The prediction is added as a
    new property to each feature, allowing you to export and validate results.

    **Input Requirements:**

    - **Feature Collection**: Must contain the same band/property names used during training
      (e.g., 'B2', 'B3', 'B4' if those were the training bands)
    - **Classifier**: Trained classifier from Learner nodes (Random Forest Learner, CART Learner,
      SVM Learner, or Naive Bayes Learner)

    **Output:**

    - Original Feature Collection with a new 'GEE_class' property containing predicted class labels
    - All original properties are preserved
    - Can be exported to table for validation

    **Common Use Cases:**

    - **Validation**: Use `Generate Training Points from Reference Image` to create new validation points,
      then use this node to predict their classes and compare with reference labels
    - **Independent Testing**: Apply trained classifier to independent test datasets
    - **Cross-Validation**: Predict on held-out validation sets
    - **Spatial Validation**: Predict on features from different regions or time periods

    **Workflow Example:**

    1. Train classifier: `Sample Regions for Classification` → `Random Forest Learner`
    2. Generate validation points: `Generate Training Points from Reference Image` (new random points)
    3. Predict: `Feature Collection Predictor` (this node)
    4. Export: `Feature Collection to Table` → Compare predictions with reference labels

    **Band/Property Matching:**

    - The Feature Collection must contain properties with the same names as the bands/features
      used during training
    - If training used 'B2', 'B3', 'B4', the Feature Collection must have these properties
    - System properties (e.g., 'system:index') are automatically excluded

    **Output Property:**

    The prediction results are stored in a property named **'GEE_class'** (automatically set).

    **Reference:**
    [Supervised Classification Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
        classifier_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get Feature Collection and classifier from connections
            feature_collection = fc_connection.feature_collection
            classifier = classifier_connection.classifier

            if not isinstance(feature_collection, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection (e.g., from 'Generate Training Points from Reference Image')"
                )

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please use a trained classifier from Learner nodes (Random Forest Learner, CART Learner, SVM Learner, or Naive Bayes Learner)."
                )

            # Classify the Feature Collection
            # Output property name is set to _ClassResult ('GEE_class')
            classified_fc = feature_collection.classify(
                classifier, outputName=_ClassResult
            )

            # Map prediction results back to original class values if reverse_mapping exists
            reverse_mapping = classifier_connection.reverse_mapping
            if reverse_mapping is not None:
                LOGGER.warning(
                    f"Mapping predictions back to original class values using reverse mapping: {reverse_mapping}"
                )

                # Build remap lists for server-side remapping
                # reverse_mapping format: {0: 11, 1: 21, 2: 22, ...}
                remap_from = list(reverse_mapping.keys())  # [0, 1, 2, ...]
                remap_to = [
                    reverse_mapping[idx] for idx in remap_from
                ]  # [11, 21, 22, ...]

                def remap_prediction(feature):
                    """Remap prediction from 0-based index to original class value"""
                    predicted_value = feature.get(_ClassResult)
                    # Use ee.Algorithms.If chain for server-side mapping
                    remapped_value = predicted_value  # Default to original value
                    for idx, orig_val in zip(remap_from, remap_to):
                        remapped_value = ee.Algorithms.If(
                            ee.Number(predicted_value).eq(ee.Number(idx)),
                            ee.Number(orig_val),
                            remapped_value,
                        )
                    return feature.set(_ClassResult, remapped_value)

                classified_fc = classified_fc.map(remap_prediction)
                LOGGER.warning(
                    "Successfully remapped predictions to original class values"
                )

            LOGGER.warning(
                f"Successfully classified Feature Collection with {_ClassResult} property"
            )

            # Log sample count if available
            try:
                fc_size = feature_collection.size().getInfo()
                LOGGER.warning(f"Classified {fc_size} features")
            except Exception:
                LOGGER.warning("Classified Feature Collection (size check skipped)")

            return knut.export_gee_feature_collection_connection(
                classified_fc, fc_connection
            )

        except Exception as e:
            LOGGER.error(f"Feature Collection classification failed: {e}")
            raise


############################################
# Classifier Scorer
############################################


@knext.node(
    name="GEE Classifier Scorer",
    node_type=knext.NodeType.OTHER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "scorer.png",
    id="classifierscorer",
    after="featurecollectionpredictor",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with classified features (from Feature Collection Predictor). Must contain both label property (true labels) and 'GEE_class' property (predicted labels).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_table(
    name="Validation Accuracy Metrics",
    description="Table containing per-class and overall classification accuracy metrics.",
)
class ClassifierScorer:
    """Computes classification accuracy metrics from classified FeatureCollection.

    This node evaluates classifier performance by computing confusion matrix-based
    metrics from a FeatureCollection that has already been classified by the
    **Feature Collection Predictor** node. It uses GEE's built-in ConfusionMatrix
    methods to compute all metrics server-side, minimizing data transfer.

    **Input Requirements:**

    - **Classified Feature Collection**: FeatureCollection output from **Feature Collection Predictor** node
    - Must contain:
      - **Label property** (e.g., 'class', 'landcover'): True class labels
      - **'GEE_class' property**: Predicted class labels (automatically added by Feature Collection Predictor)

    **Output Metrics:**

    - **Per-Class**: TP, Precision, Recall, F-measure for each class
    - **Overall**: Accuracy, Cohen's Kappa

    **Performance:**

    - Uses GEE's ConfusionMatrix methods (accuracy, kappa, producersAccuracy, consumersAccuracy)
    - All metrics computed server-side in a single operation
    - Minimal data transfer (only final metrics downloaded)

    **Workflow Example:**

    1. Split data: `Feature Collection Partitioner` → Training Set + Validation Set
    2. Train classifier: Training Set → `Random Forest Learner`
    3. Predict: Validation Set + Trained Classifier → `Feature Collection Predictor`
    4. Score: Classified Validation Set → `Classifier Scorer` (this node)
    """

    label_property = knext.StringParameter(
        "Label property",
        "Name of the property containing true class labels (e.g., 'class', 'landcover', 'LC'). This should match the label property used during training.",
        default_value="class",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        fc_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        try:
            # Get classified FeatureCollection from connection
            classified_fc = fc_connection.feature_collection

            if not isinstance(classified_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection. "
                    "Please connect the output from 'Feature Collection Predictor' node."
                )

            # Verify that the FeatureCollection contains the required properties
            try:
                first_feature = classified_fc.first()
                properties = first_feature.propertyNames().getInfo()
                if _ClassResult not in properties:
                    raise ValueError(
                        f"FeatureCollection does not contain prediction property '{_ClassResult}'. "
                        f"Please ensure the FeatureCollection has been classified by 'Feature Collection Predictor' node. "
                        f"Found properties: {properties}"
                    )
                if self.label_property not in properties:
                    raise ValueError(
                        f"FeatureCollection does not contain label property '{self.label_property}'. "
                        f"Please check the label property name. "
                        f"Found properties: {properties}"
                    )
            except Exception as e:
                LOGGER.warning(f"Could not verify properties: {e}")

            # Get data size for logging
            try:
                fc_size = classified_fc.size().getInfo()
                LOGGER.warning(
                    f"Computing validation accuracy metrics for {fc_size} classified features (label property: {self.label_property})"
                )
            except Exception:
                LOGGER.warning(
                    f"Computing validation accuracy metrics (label property: {self.label_property})"
                )

            # Compute validation accuracy metrics using GEE's built-in methods
            try:
                metrics_df = compute_validation_metrics_from_classified_fc(
                    classified_fc,
                    self.label_property,
                    prediction_property=_ClassResult,
                )
                LOGGER.warning("Successfully computed validation accuracy metrics")
            except Exception as e:
                LOGGER.error(f"Could not compute validation accuracy metrics: {e}")
                raise

            # Return the accuracy metrics table
            return knext.Table.from_pandas(metrics_df)

        except Exception as e:
            LOGGER.error(f"Classifier scoring failed: {e}")
            raise
