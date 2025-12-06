"""
GEE Unsupervised Classification (Clustering) Nodes for KNIME
This module contains nodes for unsupervised classification using Google Earth Engine Clusterer APIs.
Based on: https://developers.google.com/earth-engine/apidocs/ee-clusterer-train
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    GoogleEarthEngineObjectSpec,
    google_earth_engine_port_type,
    GEEImageConnectionObject,
    GEEFeatureCollectionConnectionObject,
    gee_image_port_type,
    gee_feature_collection_port_type,
    gee_clusterer_port_type,
)

# Category for GEE Unsupervised Classification nodes
__category = knext.category(
    path="/community/gee",
    level_id="clustering",
    name="Clustering",
    description="Google Earth Engine Unsupervised Classification (Clustering) nodes",
    icon="icons/cluster.png",
    after="supervised",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/clustering/"

# Output band name for clustering results
_ClusterResult = "cluster"

############################################
# Image Cluster Sampling
############################################


@knext.node(
    name="Image Cluster Sampling",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "SampleImagePoints.png",
    id="imageclustersampling",
    after="",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with image to sample for clustering training.",
    port_type=gee_image_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with sampled training points (band values).",
    port_type=gee_feature_collection_port_type,
)
class ImageClusterSampling:
    """Samples pixels from an image to create training data for clustering.

    This node generates random sample points from an image and extracts band values
    at these points, creating a FeatureCollection suitable for training clusterers.
    Unlike supervised classification sampling, this node does not require labels
    since clustering is unsupervised.

    **Sampling Process:**

    - Generates random sample points within the image geometry
    - Extracts band values from the image at these points
    - Output FeatureCollection contains band values as properties
    - Each feature represents one sampled pixel with all band values

    **Parameters:**

    - **Scale**: Pixel scale in meters (default: 30m, typical for Landsat/Sentinel-2)
    - **Number of Pixels**: Number of sample points to generate (default: 5000)
    - **Random Seed**: For reproducible sampling (default: 0, advanced)
    - **Tile Scale**: Performance optimization for large areas (default: 1.0, higher = faster, advanced)

    **Common Use Cases:**

    - Creating training samples for K-Means clustering
    - Generating representative samples for X-Means clustering
    - Exploratory data analysis and pattern discovery

    **Workflow:**

    1. Sample image: `Image` → This node → `Feature Collection`
    2. Train clusterer: `Feature Collection` → `K-Means Clusterer Learner` → `Clusterer`
    3. Apply clusterer: `Image` + `Clusterer` → `Apply Clusterer` → `Clustered Image`

    **Reference:**
    Based on Earth Engine clustering guide: https://developers.google.com/earth-engine/guides/clustering
    """

    scale = knext.IntParameter(
        "Scale",
        "Pixel scale in meters for sampling (e.g., 30 for Landsat, 10 for Sentinel-2)",
        default_value=30,
        min_value=1,
        max_value=1000,
    )

    num_pixels = knext.IntParameter(
        "Number of pixels",
        "Number of sample points to generate for training",
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

    tile_scale = knext.DoubleParameter(
        "Tile scale",
        "Tile scale for performance optimization (1.0 = default, higher = faster for large areas)",
        default_value=1.0,
        min_value=0.1,
        max_value=16.0,
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
            # Get image from connection
            image = image_connection.image

            if not isinstance(image, ee.Image):
                raise ValueError("Input must be an Image object")

            # Get image geometry for sampling region
            sampling_region = image.geometry()

            # Get band names
            band_names = image.bandNames().getInfo()
            LOGGER.warning(
                f"Sampling {self.num_pixels} pixels from {len(band_names)} bands: {band_names}"
            )

            # Generate random sample points from image
            LOGGER.warning(
                f"Generating {self.num_pixels} random sample points at {self.scale}m scale"
            )

            sample_points = image.sample(
                region=sampling_region,
                scale=self.scale,
                numPixels=self.num_pixels,
                seed=self.seed,
                tileScale=self.tile_scale,  # Performance optimization for large areas
                geometries=True,  # Preserve geometry for GeoDataFrame conversion
            )

            try:
                point_count = sample_points.size().getInfo()
                LOGGER.warning(f"Successfully sampled {point_count} points from image")
            except Exception:
                LOGGER.warning("Sampling completed (size check skipped)")

            return knut.export_gee_feature_collection_connection(
                sample_points, image_connection
            )

        except Exception as e:
            LOGGER.error(f"Image cluster sampling failed: {e}")
            raise


############################################
# K-Means Clusterer Learner
############################################


@knext.node(
    name="K-Means Clusterer Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "KMeans.png",
    id="kmeansclustererlearner",
    after="imageclustersampling",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection with numeric properties for clustering (no labels required).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Clusterer Connection",
    description="GEE Clusterer connection with trained K-Means model.",
    port_type=gee_clusterer_port_type,
)
class KMeansClustererLearner:
    """Trains a K-Means clusterer using training data.

    K-Means is one of the most popular and widely-used clustering algorithms.
    It partitions data into K clusters by minimizing within-cluster variance.

    **Algorithm Details:**

    - **Partitioning**: Divides data into K clusters
    - **Centroid-based**: Uses cluster centers (centroids)
    - **Fast**: Efficient for large datasets
    - **Deterministic**: Same seed produces same results

    **Parameters:**

    - **Number of Clusters (K)**: Number of clusters to create (default: 5)
    - **Initialization Method**: How to initialize cluster centers (default: random, advanced)
    - **Use Canopies**: Use canopies to reduce distance calculations (default: false, advanced)
    - **Distance Function**: Distance metric (default: Euclidean, advanced)
    - **Max Iterations**: Maximum iterations for convergence (default: null = unlimited, advanced)
    - **Preserve Order**: Preserve order of instances (default: false, advanced)
    - **Fast Mode**: Enable faster distance calculations (default: false, advanced)
    - **Random Seed**: For reproducible results (default: 10, advanced)
    - **Subsampling**: Fraction of training data to use (default: 1.0 = use all, advanced)

    **Common Use Cases:**

    - Land cover segmentation
    - Image segmentation
    - Anomaly detection
    - Data exploration
    - Preprocessing for supervised classification

    **Workflow:**

    1. Sample image: `Image` → `Image Cluster Sampling` → `Feature Collection`
    2. Train clusterer: `Feature Collection` → This node → `Clusterer`
    3. Apply clusterer: `Image` + `Clusterer` → `Apply Clusterer` → `Clustered Image`

    **Reference:**
    [Earth Engine Clustering Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for clustering (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all numeric properties. Available bands/features can be explored using the 
        **GEE Image Info Extractor** node.""",
        default_value="",
    )

    n_clusters = knext.IntParameter(
        "Number of clusters (K)",
        "Number of clusters to create",
        default_value=5,
        min_value=2,
        max_value=50,
    )

    init = knext.IntParameter(
        "Initialization method",
        "Initialization method: 0 = random, 1 = k-means++, 2 = canopy, 3 = farthest first",
        default_value=0,
        min_value=0,
        max_value=3,
        is_advanced=True,
    )

    canopies = knext.BoolParameter(
        "Use canopies",
        "Use canopies to reduce the number of distance calculations",
        default_value=False,
        is_advanced=True,
    )

    max_candidates = knext.IntParameter(
        "Max candidates",
        "Maximum number of candidate canopies to retain in memory when using canopy clustering",
        default_value=100,
        min_value=1,
        max_value=10000,
        is_advanced=True,
    )

    periodic_pruning = knext.IntParameter(
        "Periodic pruning",
        "How often to prune low density canopies when using canopy clustering",
        default_value=10000,
        min_value=1,
        max_value=100000,
        is_advanced=True,
    )

    min_density = knext.IntParameter(
        "Min density",
        "Minimum canopy density below which a canopy will be pruned during periodic pruning",
        default_value=2,
        min_value=1,
        max_value=100,
        is_advanced=True,
    )

    t1 = knext.DoubleParameter(
        "T1 distance",
        "The T1 distance to use when using canopy clustering. Value < 0 is taken as a positive multiplier for T2",
        default_value=-1.5,
        min_value=-10.0,
        max_value=10.0,
        is_advanced=True,
    )

    t2 = knext.DoubleParameter(
        "T2 distance",
        "The T2 distance to use when using canopy clustering. Values < 0 cause a heuristic based on attribute std. deviation",
        default_value=-1.0,
        min_value=-10.0,
        max_value=10.0,
        is_advanced=True,
    )

    distance_function = knext.StringParameter(
        "Distance function",
        "Distance function to use: Euclidean or Manhattan",
        default_value="Euclidean",
        enum=["Euclidean", "Manhattan"],
        is_advanced=True,
    )

    max_iterations = knext.IntParameter(
        "Max iterations",
        "Maximum number of iterations (null/unlimited if not specified)",
        default_value=0,  # 0 will be treated as None/null
        min_value=0,
        max_value=10000,
        is_advanced=True,
    )

    preserve_order = knext.BoolParameter(
        "Preserve order",
        "Preserve order of instances",
        default_value=False,
        is_advanced=True,
    )

    fast = knext.BoolParameter(
        "Fast mode",
        "Enables faster distance calculations using cut-off values. Disables calculation/output of squared errors/distances",
        default_value=False,
        is_advanced=True,
    )

    seed = knext.IntParameter(
        "Random seed",
        "The randomization seed",
        default_value=10,
        min_value=0,
        max_value=10000,
        is_advanced=True,
    )

    subsampling = knext.DoubleParameter(
        "Subsampling factor",
        "Fraction of training data to use (0.0-1.0, 1.0 = use all data)",
        default_value=1.0,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )

    subsampling_seed = knext.IntParameter(
        "Subsampling seed",
        "Random seed for subsampling",
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
                raise ValueError(
                    "Input must be a FeatureCollection with numeric properties for clustering"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(f"Training K-Means clusterer with {fc_size} samples")
            except Exception:
                LOGGER.warning(
                    "Training K-Means clusterer with FeatureCollection (size check skipped)"
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
                # Use all properties except system properties
                feature_list = available_features

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid numeric properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for clustering: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build clusterer parameters
            clusterer_params = {
                "nClusters": self.n_clusters,
                "init": self.init,
                "canopies": self.canopies,
                "seed": self.seed,
            }

            # Add optional parameters only if they differ from defaults or are explicitly set
            if self.canopies:
                clusterer_params["maxCandidates"] = self.max_candidates
                clusterer_params["periodicPruning"] = self.periodic_pruning
                clusterer_params["minDensity"] = self.min_density
                clusterer_params["t1"] = self.t1
                clusterer_params["t2"] = self.t2

            if self.distance_function != "Euclidean":
                clusterer_params["distanceFunction"] = self.distance_function

            if self.max_iterations > 0:
                clusterer_params["maxIterations"] = self.max_iterations

            if self.preserve_order:
                clusterer_params["preserveOrder"] = self.preserve_order

            if self.fast:
                clusterer_params["fast"] = self.fast

            # Create clusterer
            clusterer = ee.Clusterer.wekaKMeans(**clusterer_params)

            # Train clusterer
            train_params = {
                "features": training_fc,
                "inputProperties": feature_list if feature_list else None,
            }

            # Add subsampling if specified
            if self.subsampling < 1.0:
                train_params["subsampling"] = self.subsampling
                train_params["subsamplingSeed"] = self.subsampling_seed

            trained_clusterer = clusterer.train(**train_params)

            LOGGER.warning(
                f"Successfully trained K-Means clusterer with {self.n_clusters} clusters"
            )

            # Export clusterer connection
            clusterer_connection = knut.export_gee_clusterer_connection(
                trained_clusterer,
                training_data_connection,
                input_properties=feature_list,
            )

            return clusterer_connection

        except Exception as e:
            LOGGER.error(f"K-Means clusterer training failed: {e}")
            raise


############################################
# X-Means Clusterer Learner
############################################


@knext.node(
    name="X-Means Clusterer Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "XMeans.png",
    id="xmeansclustererlearner",
    after="kmeansclustererlearner",
)
@knext.input_port(
    name="Training Data",
    description="Training data: FeatureCollection with numeric properties for clustering (no labels required).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Clusterer Connection",
    description="GEE Clusterer connection with trained X-Means model.",
    port_type=gee_clusterer_port_type,
)
class XMeansClustererLearner:
    """Trains an X-Means clusterer that automatically determines the optimal number of clusters.

    X-Means is an extension of K-Means that automatically determines the optimal number of clusters
    within a specified range. It uses a Bayesian information criterion to decide when to split clusters.

    **Algorithm Details:**

    - **Automatic K Selection**: Determines optimal number of clusters
    - **Range-based**: Searches between min and max clusters
    - **Hierarchical**: Uses splitting criteria for refinement
    - **Adaptive**: Adjusts cluster count based on data structure

    **Parameters:**

    - **Min Clusters**: Minimum number of clusters to consider (default: 2)
    - **Max Clusters**: Maximum number of clusters to consider (default: 8)
    - **Max Iterations**: Maximum number of overall iterations (default: 3, advanced)
    - **Max K-Means**: Maximum K-means iterations per cluster count (default: 1000, advanced)
    - **Max For Children**: Maximum iterations in K-Means performed on child centers (default: 1000, advanced)
    - **Use KD-Tree**: Use a KDTree for faster neighbor searches (default: false, advanced)
    - **Cutoff Factor**: Percentage of split centroids if none of the children win (default: 0, advanced)
    - **Distance Function**: Distance metric (default: Euclidean, advanced)
    - **Random Seed**: For reproducible results (default: 10, advanced)

    **Common Use Cases:**

    - Automatic land cover segmentation
    - Unknown number of clusters scenarios
    - Exploratory data analysis
    - When optimal K is unknown

    **Workflow:**

    1. Sample image: `Image` → `Image Cluster Sampling` → `Feature Collection`
    2. Train clusterer: `Feature Collection` → This node → `Clusterer`
    3. Apply clusterer: `Image` + `Clusterer` → `Apply Clusterer` → `Clustered Image`

    **Reference:**
    [Earth Engine Clustering Guide](https://developers.google.com/earth-engine/guides/classification)
    """

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use for clustering (e.g., 'B2,B3,B4,B8'). 
        Leave empty to use all numeric properties. Available bands/features can be explored using the 
        **GEE Image Info Extractor** node.""",
        default_value="",
    )

    min_clusters = knext.IntParameter(
        "Min clusters",
        "Minimum number of clusters to consider",
        default_value=2,
        min_value=2,
        max_value=20,
    )

    max_clusters = knext.IntParameter(
        "Max clusters",
        "Maximum number of clusters to consider",
        default_value=8,
        min_value=2,
        max_value=50,
    )

    max_iterations = knext.IntParameter(
        "Max iterations",
        "Maximum number of overall iterations",
        default_value=3,
        min_value=1,
        max_value=100,
        is_advanced=True,
    )

    max_k_means = knext.IntParameter(
        "Max K-means iterations",
        "The maximum number of iterations to perform in KMeans",
        default_value=1000,
        min_value=1,
        max_value=10000,
        is_advanced=True,
    )

    max_for_children = knext.IntParameter(
        "Max for children",
        "The maximum number of iterations in KMeans that is performed on the child centers",
        default_value=1000,
        min_value=1,
        max_value=10000,
        is_advanced=True,
    )

    use_kd = knext.BoolParameter(
        "Use KD-tree",
        "Use a KDTree for faster neighbor searches",
        default_value=False,
        is_advanced=True,
    )

    cutoff_factor = knext.DoubleParameter(
        "Cutoff factor",
        "Takes the given percentage of the split centroids if none of the children win",
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        is_advanced=True,
    )

    distance_function = knext.StringParameter(
        "Distance function",
        "Distance function to use: Chebyshev, Euclidean, or Manhattan",
        default_value="Euclidean",
        enum=["Chebyshev", "Euclidean", "Manhattan"],
        is_advanced=True,
    )

    seed = knext.IntParameter(
        "Random seed",
        "The randomization seed",
        default_value=10,
        min_value=0,
        max_value=10000,
        is_advanced=True,
    )

    subsampling = knext.DoubleParameter(
        "Subsampling factor",
        "Fraction of training data to use (0.0-1.0, 1.0 = use all data)",
        default_value=1.0,
        min_value=0.01,
        max_value=1.0,
        is_advanced=True,
    )

    subsampling_seed = knext.IntParameter(
        "Subsampling seed",
        "Random seed for subsampling",
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
                raise ValueError(
                    "Input must be a FeatureCollection with numeric properties for clustering"
                )

            # Get feature count for logging
            try:
                fc_size = training_fc.size().getInfo()
                LOGGER.warning(f"Training X-Means clusterer with {fc_size} samples")
            except Exception:
                LOGGER.warning(
                    "Training X-Means clusterer with FeatureCollection (size check skipped)"
                )

            # Get available properties (similar to K-Means)
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

            # Determine feature list (same logic as K-Means)
            if self.bands:
                feature_list = [b.strip() for b in self.bands.split(",")]
                missing_bands = [b for b in feature_list if b not in available_features]
                if missing_bands:
                    LOGGER.warning(
                        f"Warning: Some specified bands not found in FeatureCollection: {missing_bands}"
                    )
                feature_list = [b for b in feature_list if b in available_features]
            else:
                feature_list = available_features

            if len(feature_list) == 0:
                raise ValueError(
                    f"No valid features found. Available properties: {available_features}. "
                    f"Please check 'Bands/Features' parameter or ensure FeatureCollection has valid numeric properties."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for clustering: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build clusterer parameters
            clusterer_params = {
                "minClusters": self.min_clusters,
                "maxClusters": self.max_clusters,
                "maxIterations": self.max_iterations,
                "maxKMeans": self.max_k_means,
                "maxForChildren": self.max_for_children,
                "useKD": self.use_kd,
                "cutoffFactor": self.cutoff_factor,
                "distanceFunction": self.distance_function,
                "seed": self.seed,
            }

            # Create clusterer
            clusterer = ee.Clusterer.wekaXMeans(**clusterer_params)

            # Train clusterer
            train_params = {
                "features": training_fc,
                "inputProperties": feature_list if feature_list else None,
            }

            # Add subsampling if specified
            if self.subsampling < 1.0:
                train_params["subsampling"] = self.subsampling
                train_params["subsamplingSeed"] = self.subsampling_seed

            trained_clusterer = clusterer.train(**train_params)

            # Get actual number of clusters found
            try:
                clusterer_info = trained_clusterer.getInfo()
                actual_clusters = clusterer_info.get(
                    "numberOfClusters", self.max_clusters
                )
                LOGGER.warning(
                    f"X-Means determined optimal number of clusters: {actual_clusters} (range: {self.min_clusters}-{self.max_clusters})"
                )
            except Exception:
                LOGGER.warning("Successfully trained X-Means clusterer")

            # Export clusterer connection
            clusterer_connection = knut.export_gee_clusterer_connection(
                trained_clusterer,
                training_data_connection,
                input_properties=feature_list,
            )

            return clusterer_connection

        except Exception as e:
            LOGGER.error(f"X-Means clusterer training failed: {e}")
            raise


############################################
# Apply Clusterer
############################################


@knext.node(
    name="Apply Clusterer",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ClusteringToImage.png",
    id="applyclusterer",
    after="xmeansclustererlearner",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with image to cluster.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="GEE Clusterer Connection",
    description="GEE Clusterer connection with trained clusterer model.",
    port_type=gee_clusterer_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with clustered image (cluster labels).",
    port_type=gee_image_port_type,
)
class ApplyClusterer:
    """Applies a trained clusterer to cluster an image.

    This node uses a trained clustering model to assign cluster labels to pixels in an image,
    producing a cluster map where each pixel is assigned to a cluster.

    **Clustering Features:**

    - **Multi-band Support**: Use all available spectral bands
    - **Cluster Labels**: Output image with cluster IDs in band named 'cluster'
    - **Preserves Geometry**: Maintains original image geometry

    **Output:**

    The node outputs an image with one band:
    - **cluster**: Cluster IDs (integer values representing cluster assignments)

    **Common Use Cases:**

    - Land cover segmentation
    - Image segmentation
    - Anomaly detection
    - Data exploration and visualization
    - Preprocessing for supervised classification

    **Workflow:**

    1. Sample image: `Image` → `Image Cluster Sampling` → `Feature Collection`
    2. Train clusterer: `Feature Collection` → `K-Means Clusterer Learner` → `Clusterer`
    3. Apply clusterer: `Image` + `Clusterer` → This node → `Clustered Image`

    **Note:** This node only works with `Image` objects. The bands/features used during clusterer training are automatically inherited from the clusterer connection.
    """

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        image_connection,
        clusterer_connection,
    ):
        import ee
        import logging

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and clusterer from connections
            image = image_connection.image
            clusterer = clusterer_connection.clusterer

            if not isinstance(clusterer, ee.Clusterer):
                raise ValueError(
                    "Clusterer connection must contain a trained Clusterer object. "
                    "Please use a trained clusterer from Clusterer Learner nodes."
                )

            # Get input properties (bands) from clusterer connection
            input_properties = clusterer_connection.input_properties
            if input_properties:
                # Select only the bands used during training
                image = image.select(input_properties)
                LOGGER.warning(
                    f"Using training bands for clustering: {input_properties}"
                )
            else:
                LOGGER.warning(
                    "No input properties found in clusterer connection, using all bands"
                )

            # Apply clusterer to image
            # Output band name is set to _ClusterResult ('cluster')
            clustered_image = image.cluster(clusterer, outputName=_ClusterResult)

            LOGGER.warning("Successfully applied clusterer to image")

            return knut.export_gee_image_connection(clustered_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Apply clusterer failed: {e}")
            raise
