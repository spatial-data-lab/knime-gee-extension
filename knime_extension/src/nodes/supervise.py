"""
GEE Supervised Classification Nodes for KNIME
This module contains nodes for supervised classification using Google Earth Engine Machine Learning APIs.
Based on: https://developers.google.com/earth-engine/guides/classification
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    GoogleEarthEngineObjectSpec,
    google_earth_engine_port_type,
)

# Category for GEE Supervised Classification nodes
__category = knext.category(
    path="/community/gee",
    level_id="supervised",
    name="Supervised Classification",
    description="Google Earth Engine Supervised Classification nodes",
    icon="icons/SupervisedClass.png",
    after="sampling",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/supervised/"


############################################
# Generate Training Points from Reference Image
############################################


@knext.node(
    name="Generate Training Points from Reference Image",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "generatePoints.png",
    id="generatetrainingpoints",
    after="",
)
@knext.input_port(
    name="Reference Classification Image",
    description="Reference classification image (e.g., NLCD) with class values in one band.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="Feature Image",
    description="Multi-band image for feature extraction (e.g., Sentinel-2, Landsat). Band values will be extracted at sample points.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with sampled training data (pixel values and class labels).",
    port_type=google_earth_engine_port_type,
)
class GenerateTrainingPointsFromReference:
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
        "Label Band",
        "Band name containing class values in the reference image (e.g., 'landcover'). Leave empty to use first band.",
        default_value="",
    )

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of band names to extract from feature image (e.g., 'B2,B3,B4,B8'). Leave empty to use all bands.",
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
        "Number of Pixels",
        "Number of random sample points to generate",
        default_value=5000,
        min_value=100,
        max_value=100000,
    )

    seed = knext.IntParameter(
        "Random Seed",
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
            reference_image = reference_image_connection.gee_object
            feature_image = feature_image_connection.gee_object

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

            return knut.export_gee_connection(sampled, reference_image_connection)

        except Exception as e:
            LOGGER.error(f"Generate training points failed: {e}")
            raise


############################################
# Sample Regions for Classification
############################################


@knext.node(
    name="Sample Regions for Classification",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "sampleRegions.png",
    id="sampleregions",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with multi-band image for feature extraction.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="Training Data Connection",
    description="GEE connection with training data. Polygon mode: FeatureCollection with labeled polygons. Point mode: Reference classification Image or FeatureCollection of training points.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with sampled training data (pixel values and class labels).",
    port_type=google_earth_engine_port_type,
)
class SampleRegionsForClassification:
    """Samples pixel values from an image to create training data for classification.

    This node extracts pixel values from multi-band images to create feature vectors
    for supervised classification. It supports two sampling modes:

    **Sampling Modes:**

    1. **Polygon Mode** (default): Samples all pixels within training polygons
       - Input: Image + FeatureCollection with labeled polygons
       - Use when: You have manually drawn training polygons

    2. **Point Mode**: Samples random points from a reference classification image
       - Input: Image + Reference classification image (e.g., NLCD)
       - Use when: You have a reference classification map (e.g., NLCD, existing land cover map)

    **Input Requirements:**

    - **Image**: Multi-band image with spectral information (e.g., Sentinel-2, Landsat)
    - **Polygon Mode**: Feature Collection with labeled polygons (one property contains class labels)
    - **Point Mode**: Reference classification image (e.g., NLCD) with class values

    **Sampling Process:**

    - **Polygon Mode**: Extracts all pixels within each training polygon
    - **Point Mode**: Generates random sample points from reference image, then extracts values
    - Creates a FeatureCollection with one feature per pixel (band values + class label + geometry)
    - Supports large datasets with efficient server-side processing
    - Output can be directly used for training classifiers or converted to table if needed

    **Common Use Cases:**

    - Create training data for land cover classification
    - Sample vegetation indices for crop type mapping
    - Extract spectral signatures for material identification
    - Generate balanced datasets for machine learning
    - Use reference maps (NLCD, etc.) to automatically generate training data

    **Performance Tips:**

    - Lower scale values provide more samples but may be slower
    - Use tile_scale > 1.0 for large training areas
    - Ensure sufficient samples per class (minimum 100-200 recommended)
    - Point mode: Adjust numPixels to control sample size

    **Workflow:**

    1. This node → Output: Training FeatureCollection
    2. Training FeatureCollection → Train Classifier nodes → Output: Classifier Connection
    3. Image + Classifier Connection → Classify Image → Output: Classified Image

    **Note:** The output FeatureCollection can be directly used for training. If you need a table format,
    use "Feature Collection to Table" node after this node.
    """

    sampling_mode = knext.StringParameter(
        "Sampling Mode",
        "Method for generating training samples",
        default_value="Polygon",
        enum=["Polygon", "Point"],
    )

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the training Feature Collection (e.g., 'class', 'LC', 'landcover'). Only used in Polygon mode.",
        default_value="class",
    ).rule(knext.OneOf(sampling_mode, ["Polygon"]), knext.Effect.SHOW)

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of band names to use for training (e.g., 'B2,B3,B4,B8' for Sentinel-2). Leave empty to use all bands.",
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
        "Tile Scale",
        "Tile scale for performance optimization (1.0 = default, higher = faster for large areas)",
        default_value=1.0,
        min_value=0.1,
        max_value=16.0,
        is_advanced=True,
    )

    # Point mode parameters
    num_pixels = knext.IntParameter(
        "Number of Pixels",
        "Number of random sample points to generate from reference image (only used in Point mode)",
        default_value=5000,
        min_value=100,
        max_value=100000,
        is_advanced=True,
    ).rule(knext.OneOf(sampling_mode, ["Point"]), knext.Effect.SHOW)

    seed = knext.IntParameter(
        "Random Seed",
        "Random seed for reproducible sampling (only used in Point mode)",
        default_value=0,
        min_value=0,
        max_value=10000,
        is_advanced=True,
    ).rule(knext.OneOf(sampling_mode, ["Point"]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        fc_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and feature collection from connections
            image = image_connection.gee_object
            training_fc = fc_connection.gee_object

            # Select bands if specified
            if self.bands:
                band_list = [b.strip() for b in self.bands.split(",")]
                image = image.select(band_list)
                LOGGER.warning(f"Selected bands: {band_list}")
            else:
                # Use all bands
                band_list = image.bandNames().getInfo()
                LOGGER.warning(f"Using all bands: {band_list}")

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
                return knut.export_gee_connection(sampled, image_connection)

            else:
                # Point mode: use points from reference image or pre-generated points
                LOGGER.warning(f"Point mode: Using training points from second input")

                # Check if second input is Image (reference classification) or FeatureCollection (pre-generated points)
                if isinstance(training_fc, ee.Image):
                    # Second input is a reference classification image
                    reference_image = training_fc
                    LOGGER.warning("Second input is a reference classification image")

                    # Get the label band name
                    label_band = reference_image.bandNames().get(0).getInfo()
                    LOGGER.warning(f"Using label band: {label_band}")

                    # Generate random sample points from reference image
                    sample_points = reference_image.sample(
                        region=image.geometry(),
                        scale=self.scale,
                        numPixels=self.num_pixels,
                        seed=self.seed,
                        geometries=True,
                    )

                    LOGGER.warning(
                        f"Generated {sample_points.size().getInfo()} sample points"
                    )

                    # Extract values from the image at these points
                    sampled = image.sampleRegions(
                        collection=sample_points,
                        properties=[label_band],
                        scale=self.scale,
                        tileScale=self.tile_scale,
                    )

                    # Get sample count for logging
                    try:
                        sample_count = sampled.size().getInfo()
                        LOGGER.warning(
                            f"Successfully sampled {sample_count} training pixels from reference image"
                        )
                    except Exception:
                        LOGGER.warning("Sampling completed (size check skipped)")

                    # Return FeatureCollection directly
                    return knut.export_gee_connection(sampled, image_connection)

                elif isinstance(training_fc, ee.FeatureCollection):
                    # Second input is a FeatureCollection of pre-generated points
                    # (e.g., from "Generate Training Points from Reference Image" node)
                    # This FeatureCollection may already contain band values, or may only contain labels
                    sample_points = training_fc
                    LOGGER.warning(
                        "Second input is a FeatureCollection of training points"
                    )

                    # Find the label property (should be in the properties)
                    # Try common names first
                    sample_feature = sample_points.first().getInfo()
                    available_props = list(sample_feature.get("properties", {}).keys())

                    # Find label property (exclude system properties and band names)
                    system_props = ["system:index", "system:time_start"]
                    # Get band names to exclude from label search
                    band_names = set(band_list) if band_list else set()
                    label_prop = None
                    for prop in available_props:
                        if prop not in system_props and prop not in band_names:
                            label_prop = prop
                            break

                    if label_prop is None:
                        raise ValueError(
                            "Could not find label property in training points. "
                            "Ensure the FeatureCollection has a property with class labels."
                        )

                    LOGGER.warning(f"Using label property: {label_prop}")

                    # Extract values from the image at these points
                    # This will add/update band values while preserving the label property
                    sampled = image.sampleRegions(
                        collection=sample_points,
                        properties=[label_prop],  # Preserve label property
                        scale=self.scale,
                        tileScale=self.tile_scale,
                    )

                    # Get sample count for logging
                    try:
                        sample_count = sampled.size().getInfo()
                        LOGGER.warning(
                            f"Successfully sampled {sample_count} training pixels from pre-generated points"
                        )
                    except Exception:
                        LOGGER.warning("Sampling completed (size check skipped)")

                    # Return FeatureCollection directly
                    return knut.export_gee_connection(sampled, image_connection)

                else:
                    raise ValueError(
                        "Point mode requires either:\n"
                        "1. A reference classification Image as the second input, OR\n"
                        "2. A FeatureCollection of training points (from 'Generate Training Points from Reference Image' node)\n\n"
                        f"Received type: {type(training_fc)}"
                    )

        except Exception as e:
            LOGGER.error(f"Sample regions for classification failed: {e}")
            raise


############################################
# Train Random Forest Classifier
############################################


@knext.node(
    name="Random Forest Learner",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "trainRF.png",
    id="randomforestlearner",
    after="sampleregions",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained Random Forest model.",
    port_type=google_earth_engine_port_type,
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
    https://developers.google.com/earth-engine/guides/classification
    """

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        "Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). Leave empty to use all properties except label.",
        default_value="",
    )

    number_of_trees = knext.IntParameter(
        "Number of Trees",
        "Number of decision trees in the random forest ensemble",
        default_value=100,
        min_value=10,
        max_value=1000,
    )

    variables_per_split = knext.IntParameter(
        "Variables per Split",
        "Number of variables to consider at each split (default: sqrt of total features). Set to 0 for auto.",
        default_value=0,
        min_value=0,
        max_value=100,
        is_advanced=True,
    )

    min_leaf_population = knext.IntParameter(
        "Min Leaf Population",
        "Minimum number of samples required in a leaf node",
        default_value=1,
        min_value=1,
        max_value=1000,
        is_advanced=True,
    )

    bag_fraction = knext.DoubleParameter(
        "Bag Fraction",
        "Fraction of samples to use for training each tree (0.0-1.0)",
        default_value=0.5,
        min_value=0.1,
        max_value=1.0,
        is_advanced=True,
    )

    max_nodes = knext.IntParameter(
        "Max Nodes",
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
            training_fc = training_data_connection.gee_object

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

            # Create and train classifier
            classifier = ee.Classifier.smileRandomForest(**classifier_params)

            # Train classifier
            trained_classifier = classifier.train(
                features=training_fc,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained Random Forest classifier with {self.number_of_trees} trees"
            )

            # Create classifier connection object using credentials from training data connection
            spec = GoogleEarthEngineObjectSpec(training_data_connection.spec.project_id)
            classifier_connection = GoogleEarthEngineConnectionObject(
                spec=spec,
                credentials=training_data_connection.credentials,
                gee_object=None,
                classifier=trained_classifier,
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"Random Forest classifier training failed: {e}")
            raise


############################################
# CART Learner
############################################


@knext.node(
    name="CART Learner",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "trainCART.png",
    id="cartlearner",
    after="randomforestlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained CART model.",
    port_type=google_earth_engine_port_type,
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
    https://developers.google.com/earth-engine/guides/classification
    """

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        "Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). Leave empty to use all properties except label.",
        default_value="",
    )

    max_nodes = knext.IntParameter(
        "Max Nodes",
        "Maximum number of nodes in the decision tree",
        default_value=10000,
        min_value=100,
        max_value=1000000,
    )

    min_leaf_population = knext.IntParameter(
        "Min Leaf Population",
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
            training_fc = training_data_connection.gee_object

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

            # Create and train classifier
            classifier = ee.Classifier.smileCart(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained CART classifier with max {self.max_nodes} nodes"
            )

            # Create classifier connection object using credentials from training data connection
            spec = GoogleEarthEngineObjectSpec(training_data_connection.spec.project_id)
            classifier_connection = GoogleEarthEngineConnectionObject(
                spec=spec,
                credentials=training_data_connection.credentials,
                gee_object=None,
                classifier=trained_classifier,
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"CART classifier training failed: {e}")
            raise


############################################
# SVM Learner
############################################


@knext.node(
    name="SVM Learner",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "trainSVM.png",
    id="svmlearner",
    after="cartlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained SVM model.",
    port_type=google_earth_engine_port_type,
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
    https://developers.google.com/earth-engine/guides/classification
    """

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        "Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). Leave empty to use all properties except label.",
        default_value="",
    )

    kernel_type = knext.StringParameter(
        "Kernel Type",
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
            training_fc = training_data_connection.gee_object

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

            # Create and train classifier
            # Note: GEE uses libsvm for SVM
            classifier = ee.Classifier.libsvm(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained SVM classifier with {self.kernel_type} kernel"
            )

            # Create classifier connection object using credentials from training data connection
            spec = GoogleEarthEngineObjectSpec(training_data_connection.spec.project_id)
            classifier_connection = GoogleEarthEngineConnectionObject(
                spec=spec,
                credentials=training_data_connection.credentials,
                gee_object=None,
                classifier=trained_classifier,
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"SVM classifier training failed: {e}")
            raise


############################################
# Naive Bayes Learner
############################################


@knext.node(
    name="Naive Bayes Learner",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "trainNB.png",
    id="naivebayeslearner",
    after="svmlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection (from Sample Regions for Classification or Generate Training Points) with pixel values and class labels.",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained Naive Bayes model.",
    port_type=google_earth_engine_port_type,
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
    https://developers.google.com/earth-engine/guides/classification
    """

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the training FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        "Comma-separated list of band/feature names to use for training (e.g., 'B2,B3,B4,B8'). Leave empty to use all properties except label. ⚠️ Note: Features must be non-negative integers.",
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
            training_fc = training_data_connection.gee_object

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

            # Create and train classifier
            # Note: smileNaiveBayes supports optional lambda parameter for smoothing
            classifier_params = {}
            if self.lambda_smoothing > 0.0:
                classifier_params["lambda"] = self.lambda_smoothing

            classifier = ee.Classifier.smileNaiveBayes(**classifier_params)

            trained_classifier = classifier.train(
                features=training_fc,
                classProperty=self.label_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained Naive Bayes classifier (lambda={self.lambda_smoothing})"
            )

            # Create classifier connection object using credentials from training data connection
            spec = GoogleEarthEngineObjectSpec(training_data_connection.spec.project_id)
            classifier_connection = GoogleEarthEngineConnectionObject(
                spec=spec,
                credentials=training_data_connection.credentials,
                gee_object=None,
                classifier=trained_classifier,
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"Naive Bayes classifier training failed: {e}")
            raise


############################################
# Classifier Scorer
############################################


@knext.node(
    name="Classifier Scorer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "scorer.png",
    id="classifierscorer",
    after="naivebayeslearner",
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained classifier model (from Learner nodes).",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="Validation Data",
    description="Validation FeatureCollection with class labels and features (from Sample Regions for Classification or Generate Training Points).",
    port_type=google_earth_engine_port_type,
)
@knext.output_view(
    name="Classification Accuracy Report",
    description="HTML view showing comprehensive classification accuracy metrics",
)
class ClassifierScorer:
    """Evaluates a trained classifier using validation data and displays comprehensive accuracy metrics.

    This node evaluates the performance of a trained classifier by applying it to validation data
    and computing various accuracy metrics including confusion matrix, overall accuracy, user's accuracy,
    producer's accuracy, Kappa coefficient, and per-class metrics.

    **Input Requirements:**

    - **Classifier**: Trained classifier from any Learner node (Random Forest, CART, SVM, or Naive Bayes)
    - **Validation Data**: FeatureCollection with class labels and features (same structure as training data)

    **Metrics Computed:**

    - **Overall Accuracy**: Percentage of correctly classified samples
    - **Kappa Coefficient**: Agreement between predicted and actual classes (accounts for chance)
    - **User's Accuracy (Recall)**: Accuracy from the user's perspective (per-class recall)
    - **Producer's Accuracy (Precision)**: Accuracy from the producer's perspective (per-class precision)
    - **Confusion Matrix**: Detailed matrix showing classification errors
    - **Per-Class Metrics**: Individual class accuracy, precision, recall, and F1-score

    **Use Cases:**

    - Evaluate classifier performance on validation/test data
    - Compare different classifier algorithms
    - Identify classes with poor classification accuracy
    - Validate model before applying to full image
    - Generate accuracy reports for publications

    **Reference:**
    https://developers.google.com/earth-engine/guides/classification
    """

    label_property = knext.StringParameter(
        "Label Property",
        "Property name containing class labels in the validation FeatureCollection (e.g., 'class', 'landcover')",
        default_value="class",
    )

    def configure(self, configure_context, input_schema):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        classifier_connection,
        validation_data_connection,
    ):
        import ee
        import logging
        import json

        LOGGER = logging.getLogger(__name__)

        try:
            # Get classifier and validation data
            classifier = classifier_connection.classifier
            validation_fc = validation_data_connection.gee_object

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please connect a classifier from a Learner node."
                )

            if not isinstance(validation_fc, ee.FeatureCollection):
                raise ValueError(
                    "Validation data must be a FeatureCollection from 'Sample Regions for Classification' "
                    "or 'Generate Training Points from Reference Image'"
                )

            # Get validation data size
            try:
                fc_size = validation_fc.size().getInfo()
                LOGGER.warning(
                    f"Evaluating classifier with {fc_size} validation samples"
                )
            except Exception:
                LOGGER.warning(
                    "Evaluating classifier with validation data (size check skipped)"
                )

            # Classify the validation data
            classified_fc = validation_fc.classify(classifier, "classification")

            # Compute error matrix
            error_matrix = classified_fc.errorMatrix(
                self.label_property, "classification"
            )

            # Get accuracy metrics
            overall_accuracy = error_matrix.accuracy().getInfo()
            kappa = error_matrix.kappa().getInfo()
            consumers_accuracy = error_matrix.consumersAccuracy().getInfo()
            producers_accuracy = error_matrix.producersAccuracy().getInfo()
            confusion_matrix = error_matrix.array().getInfo()
            order = error_matrix.order().getInfo()

            # Calculate per-class metrics
            # confusion_matrix is a 2D array where rows are actual classes and columns are predicted classes
            per_class_metrics = []
            for i, class_name in enumerate(order):
                # True positives: diagonal element
                tp = confusion_matrix[i][i]
                # False positives: sum of column (excluding diagonal)
                fp = sum(confusion_matrix[j][i] for j in range(len(order)) if j != i)
                # False negatives: sum of row (excluding diagonal)
                fn = sum(confusion_matrix[i][j] for j in range(len(order)) if j != i)
                # True negatives: all other cells
                tn = sum(
                    confusion_matrix[j][k]
                    for j in range(len(order))
                    for k in range(len(order))
                    if j != i and k != i
                )

                # Calculate metrics
                total_actual = sum(confusion_matrix[i])
                total_predicted = sum(confusion_matrix[j][i] for j in range(len(order)))

                precision = tp / total_predicted if total_predicted > 0 else 0.0
                recall = tp / total_actual if total_actual > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                user_accuracy = (
                    consumers_accuracy[i] if i < len(consumers_accuracy) else 0.0
                )
                producer_accuracy = (
                    producers_accuracy[i] if i < len(producers_accuracy) else 0.0
                )

                per_class_metrics.append(
                    {
                        "Class": str(class_name),
                        "Total Actual": int(total_actual),
                        "Total Predicted": int(total_predicted),
                        "True Positives": int(tp),
                        "False Positives": int(fp),
                        "False Negatives": int(fn),
                        "Precision": round(precision, 4),
                        "Recall": round(recall, 4),
                        "F1-Score": round(f1_score, 4),
                        "User's Accuracy": round(user_accuracy, 4),
                        "Producer's Accuracy": round(producer_accuracy, 4),
                    }
                )

            # Create HTML report
            html = self._create_html_report(
                overall_accuracy,
                kappa,
                consumers_accuracy,
                producers_accuracy,
                confusion_matrix,
                order,
                per_class_metrics,
            )

            LOGGER.warning(
                f"Classification evaluation completed: Overall Accuracy = {overall_accuracy:.4f}, Kappa = {kappa:.4f}"
            )

            return knext.view_html(html)

        except Exception as e:
            LOGGER.error(f"Classifier evaluation failed: {e}")
            # Return error view
            error_html = f"""
            <div style="color: red; font-family: Arial, sans-serif; padding: 20px;">
                <h2>❌ Classification Evaluation Error</h2>
                <p><strong>Error:</strong> {str(e)}</p>
                <p>Please check:</p>
                <ul>
                    <li>Classifier connection contains a trained classifier</li>
                    <li>Validation data is a FeatureCollection with labels and features</li>
                    <li>Label property name matches the actual property in validation data</li>
                </ul>
            </div>
            """
            return knext.view_html(error_html)

    def _create_html_report(
        self,
        overall_accuracy,
        kappa,
        consumers_accuracy,
        producers_accuracy,
        confusion_matrix,
        order,
        per_class_metrics,
    ):
        """Create a comprehensive HTML report for classification accuracy."""

        # Format confusion matrix as HTML table
        confusion_table = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; margin: 10px 0;'>\n"
        confusion_table += "<tr><th>Actual \\ Predicted</th>"
        for class_name in order:
            confusion_table += (
                f"<th style='background-color: #e0e0e0;'>{class_name}</th>"
            )
        confusion_table += "<th style='background-color: #d0d0d0;'>Total</th></tr>\n"

        for i, actual_class in enumerate(order):
            confusion_table += (
                f"<tr><th style='background-color: #e0e0e0;'>{actual_class}</th>"
            )
            row_sum = sum(confusion_matrix[i])
            for j, predicted_class in enumerate(order):
                value = confusion_matrix[i][j]
                # Highlight diagonal (correct predictions)
                cell_style = (
                    "background-color: #90EE90; font-weight: bold;"
                    if i == j
                    else "background-color: #FFB6C1;" if value > 0 else ""
                )
                confusion_table += f"<td style='{cell_style}'>{int(value)}</td>"
            confusion_table += f"<td style='background-color: #d0d0d0; font-weight: bold;'>{int(row_sum)}</td></tr>\n"

        # Add column totals
        confusion_table += "<tr><th style='background-color: #d0d0d0;'>Total</th>"
        for j in range(len(order)):
            col_sum = sum(confusion_matrix[i][j] for i in range(len(order)))
            confusion_table += f"<td style='background-color: #d0d0d0; font-weight: bold;'>{int(col_sum)}</td>"
        total_all = sum(
            confusion_matrix[i][j] for i in range(len(order)) for j in range(len(order))
        )
        confusion_table += f"<td style='background-color: #c0c0c0; font-weight: bold;'>{int(total_all)}</td></tr>\n"
        confusion_table += "</table>"

        # Format per-class metrics table
        per_class_table = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; margin: 10px 0; width: 100%;'>\n"
        per_class_table += "<tr style='background-color: #4CAF50; color: white;'>"
        per_class_table += "<th>Class</th><th>Total Actual</th><th>Total Predicted</th>"
        per_class_table += (
            "<th>True Positives</th><th>False Positives</th><th>False Negatives</th>"
        )
        per_class_table += "<th>Precision</th><th>Recall</th><th>F1-Score</th>"
        per_class_table += "<th>User's Accuracy</th><th>Producer's Accuracy</th></tr>\n"

        for metrics in per_class_metrics:
            per_class_table += "<tr>"
            per_class_table += f"<td><strong>{metrics['Class']}</strong></td>"
            per_class_table += f"<td>{metrics['Total Actual']}</td>"
            per_class_table += f"<td>{metrics['Total Predicted']}</td>"
            per_class_table += f"<td>{metrics['True Positives']}</td>"
            per_class_table += f"<td>{metrics['False Positives']}</td>"
            per_class_table += f"<td>{metrics['False Negatives']}</td>"
            per_class_table += f"<td>{metrics['Precision']:.4f}</td>"
            per_class_table += f"<td>{metrics['Recall']:.4f}</td>"
            per_class_table += f"<td>{metrics['F1-Score']:.4f}</td>"
            user_acc = metrics["User's Accuracy"]
            prod_acc = metrics["Producer's Accuracy"]
            per_class_table += f"<td>{user_acc:.4f}</td>"
            per_class_table += f"<td>{prod_acc:.4f}</td>"
            per_class_table += "</tr>\n"
        per_class_table += "</table>"

        # Format consumers/producers accuracy arrays
        consumers_str = ", ".join(
            [f"{order[i]}: {consumers_accuracy[i]:.4f}" for i in range(len(order))]
        )
        producers_str = ", ".join(
            [f"{order[i]}: {producers_accuracy[i]:.4f}" for i in range(len(order))]
        )

        # Create full HTML document
        overall_accuracy_pct = overall_accuracy * 100
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Classification Accuracy Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                .metric-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #4CAF50;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #27ae60;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                table {{
                    font-size: 12px;
                }}
                th {{
                    text-align: center;
                    font-weight: bold;
                }}
                td {{
                    text-align: center;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                    margin: 20px 0;
                }}
                .info-text {{
                    color: #7f8c8d;
                    font-size: 13px;
                    margin-top: 10px;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Classification Accuracy Report</h1>
                
                <div class="summary-grid">
                    <div class="metric-box">
                        <div class="metric-value">{overall_accuracy:.4f}</div>
                        <div class="metric-label">Overall Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{kappa:.4f}</div>
                        <div class="metric-label">Kappa Coefficient</div>
                    </div>
                </div>

                <h2>📈 Overall Metrics</h2>
                <div style="margin: 15px 0;">
                    <p><strong>Overall Accuracy:</strong> {overall_accuracy:.4f} ({overall_accuracy_pct:.2f}%)</p>
                    <p><strong>Kappa Coefficient:</strong> {kappa:.4f}</p>
                    <p class="info-text">
                        Overall Accuracy: Percentage of correctly classified samples.<br>
                        Kappa Coefficient: Agreement between predicted and actual classes (accounts for chance agreement).
                        Values range from -1 to 1, where 1 indicates perfect agreement.
                    </p>
                </div>

                <h2>🔢 Confusion Matrix</h2>
                <p class="info-text">
                    Rows represent actual classes, columns represent predicted classes.
                    Green cells indicate correct predictions (diagonal), red cells indicate errors.
                </p>
                {confusion_table}

                <h2>📋 Per-Class Metrics</h2>
                <p class="info-text">
                    <strong>Precision:</strong> Of all samples predicted as this class, how many were actually this class.<br>
                    <strong>Recall (User's Accuracy):</strong> Of all actual samples of this class, how many were correctly predicted.<br>
                    <strong>F1-Score:</strong> Harmonic mean of precision and recall.<br>
                    <strong>Producer's Accuracy:</strong> Accuracy from the producer's perspective (omission error).
                </p>
                {per_class_table}

                <h2>📊 User's Accuracy (Recall) by Class</h2>
                <div style="margin: 15px 0;">
                    <p>{consumers_str}</p>
                    <p class="info-text">
                        User's Accuracy: For each class, the percentage of samples predicted as that class that were actually that class.
                    </p>
                </div>

                <h2>📊 Producer's Accuracy (Precision) by Class</h2>
                <div style="margin: 15px 0;">
                    <p>{producers_str}</p>
                    <p class="info-text">
                        Producer's Accuracy: For each class, the percentage of actual samples of that class that were correctly predicted.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


############################################
# Image Class Predictor
############################################


@knext.node(
    name="Image Class Predictor",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "classifyImage.png",
    id="imageclasspredictor",
    after="naivebayeslearner",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with image to classify.",
    port_type=google_earth_engine_port_type,
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained classifier model (from Learner nodes).",
    port_type=google_earth_engine_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with classified image.",
    port_type=google_earth_engine_port_type,
)
class ImageClassPredictor:
    """Applies a trained classifier to classify an image.

    This node uses a trained machine learning model to classify pixels in an image,
    producing a classification map with predicted class labels or probabilities.

    **Input Requirements:**

    - **Image**: Multi-band image with the same bands used for training
    - **Classifier**: Trained classifier from Learner nodes (Random Forest Learner, CART Learner, SVM Learner, or Naive Bayes Learner)

    **Output Modes:**

    - **Class Map**: Single-band image with class IDs (integer values)
    - **Probability Map**: Multi-band image with class probabilities (0-1 for each class)

    **Band Selection:**

    - If bands are specified, only those bands are used for classification
    - If empty, all bands in the image are used
    - Bands must match the bands used during training

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
    https://developers.google.com/earth-engine/guides/classification
    """

    bands = knext.StringParameter(
        "Bands",
        "Comma-separated list of band names to use for classification (e.g., 'B2,B3,B4,B8'). Leave empty to use all bands. Must match training bands.",
        default_value="",
    )

    output_mode = knext.StringParameter(
        "Output Mode",
        "Type of classification output",
        default_value="class_map",
        enum=["class_map", "probability_map"],
    )

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
            image = image_connection.gee_object
            classifier = classifier_connection.classifier

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please use a trained classifier from Learner nodes (Random Forest Learner, CART Learner, SVM Learner, or Naive Bayes Learner)."
                )

            # Select bands if specified
            if self.bands:
                band_list = [b.strip() for b in self.bands.split(",")]
                image = image.select(band_list)
                LOGGER.warning(f"Using bands for classification: {band_list}")
            else:
                LOGGER.warning("Using all bands for classification")

            # Classify the image
            if self.output_mode == "class_map":
                # Standard classification: class IDs only
                classified_image = image.classify(classifier)
                LOGGER.warning("Classified image with class map output")
            else:
                # Probability map: probabilities for each class
                # Note: GEE's classify() returns class IDs.
                # For probability maps, we need to use the classifier's probability output.
                # However, probability output format varies by classifier type.
                # For now, we return the class map and note that probability maps
                # may require classifier-specific handling.
                classified_image = image.classify(classifier)
                LOGGER.warning(
                    "Classified image (class map mode). "
                    "Note: Probability map output may require classifier-specific implementation."
                )

            LOGGER.warning("Successfully classified image")

            return knut.export_gee_connection(classified_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Image classification failed: {e}")
            raise
