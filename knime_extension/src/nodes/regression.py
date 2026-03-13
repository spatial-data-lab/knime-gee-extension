"""
GEE Regression nodes for KNIME.
Regression analysis: linear and multilinear fit, CART regression, prediction on images and feature collections.
"""

import knime.extension as knext
import util.knime_utils as knut
from util.common import (
    GoogleEarthEngineConnectionObject,
    google_earth_engine_port_type,
    gee_image_port_type,
    gee_feature_collection_port_type,
    gee_classifier_port_type,
)

# Category for GEE Regression nodes
__category = knext.category(
    path="/community/gee",
    level_id="regression",
    name="GEE Regression",
    description="Google Earth Engine Regression Analysis nodes",
    icon="icons/Regression.png",  # Reuse classification icon or create new one
    after="supervised",
)

# Node icon path
__NODE_ICON_PATH = "icons/icon/regression/"  # Reuse supervised icons or create new ones


############################################
# Linear Fit
############################################


@knext.node(
    name="GEE Linear Fit",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "LinearFit.png",
    id="linearfit",
    after="",
)
@knext.input_port(
    name="Training Image",
    description="GEE Image connection with two bands: independent variable (first) and dependent variable (second).",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Region",
    description="GEE Feature Collection connection with geometry for regression training area.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with predicted values (if prediction enabled).",
    port_type=gee_image_port_type,
)
@knext.output_table(
    name="Regression Coefficients",
    description="Table containing regression coefficients (offset/intercept and scale/slope).",
)
class LinearFit:
    """Performs simple linear regression using ee.Reducer.linearFit.

    This node performs a least squares linear regression with one independent variable
    and one dependent variable. The regression equation is: Y = α + βX + ε,
    where α is the intercept (offset) and β is the slope (scale).

    **Input Requirements:**

    - Training image must have exactly 2 bands:
      - First band: Independent variable (X)
      - Second band: Dependent variable (Y)
    - Region: FeatureCollection or Geometry defining the training area

    **Output:**

    - Regression coefficients: intercept (offset) and slope (scale)
    - Optional: Predicted image with regression applied

    **Common Use Cases:**

    - Estimate tree cover percentage from NDVI
    - Predict biomass from vegetation indices
    - Model relationships between spectral bands
    - Create continuous value predictions from indices

    **Note:** For large regions, use bestEffort to avoid maxPixels errors.
    """

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        max_value=1000,
        scale_description="Pixel resolution in meters for the regression calculation (e.g., 30 for Landsat, 10 for Sentinel-2). Only used when Use NominalScale is disabled.",
    )

    best_effort = knext.BoolParameter(
        "Best effort",
        "If enabled, automatically adjusts scale to avoid maxPixels errors for large regions.",
        default_value=True,
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=knut.GEE_MAX_PIXELS,
        description="Maximum number of pixels to use (only used if bestEffort is disabled).",
        is_advanced=True,
    ).rule(knext.OneOf(best_effort, [False]), knext.Effect.SHOW)

    generate_prediction = knext.BoolParameter(
        "Generate prediction image",
        "If enabled, applies the regression model to the independent variable band to create a prediction image.",
        default_value=False,
    )

    predicted_band_name = knext.StringParameter(
        "Predicted band name",
        "Name for the output prediction band (only used if prediction is enabled).",
        default_value="predicted",
    ).rule(knext.OneOf(generate_prediction, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_image_connection,
        region_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and region from connections
            training_image = training_image_connection.image
            region = region_connection.feature_collection

            # Validate that image has 2 bands
            band_names = training_image.bandNames().getInfo()
            if len(band_names) != 2:
                raise ValueError(
                    f"Training image must have exactly 2 bands (independent variable, dependent variable). "
                    f"Found {len(band_names)} bands: {band_names}"
                )

            # Get geometry from FeatureCollection
            if isinstance(region, ee.FeatureCollection):
                geometry = region.geometry()
            else:
                geometry = region

            scale_value = knut.resolve_scale(
                self.use_nominal_scale, self.scale, training_image
            )

            # Prepare reduceRegion parameters
            reduce_params = {
                "reducer": ee.Reducer.linearFit(),
                "geometry": geometry,
                "scale": scale_value,
            }

            if self.best_effort:
                reduce_params["bestEffort"] = True
            else:
                reduce_params["maxPixels"] = self.max_pixels

            # Perform linear fit
            linear_fit_result = training_image.reduceRegion(**reduce_params)

            # Get coefficients
            offset = linear_fit_result.get("offset").getInfo()
            scale_coef = linear_fit_result.get("scale").getInfo()

            LOGGER.warning(
                f"Linear fit completed: intercept (offset) = {offset:.6f}, slope (scale) = {scale_coef:.6f}"
            )

            # Create coefficients table without description column
            coefficients_df = pd.DataFrame(
                {
                    "coefficient": ["intercept", "slope"],
                    "value": [offset, scale_coef],
                }
            )

            # Generate prediction image if requested
            output_image = training_image
            if self.generate_prediction:
                # Always use first band as independent variable (same as training)
                independent_band = training_image.select([0])

                # Apply regression: predicted = intercept + slope * independent
                predicted = independent_band.expression(
                    "intercept + slope * x",
                    {
                        "x": independent_band.select([0]),
                        "intercept": ee.Number(offset),
                        "slope": ee.Number(scale_coef),
                    },
                ).rename(self.predicted_band_name)

                output_image = predicted
                LOGGER.warning(
                    f"Prediction image generated with band name: {self.predicted_band_name}"
                )

            return (
                knut.export_gee_image_connection(
                    output_image, training_image_connection
                ),
                knext.Table.from_pandas(coefficients_df),
            )

        except Exception as e:
            LOGGER.error(f"Linear fit failed: {e}")
            raise


############################################
# Linear Regression
############################################


@knext.node(
    name="GEE Linear Regression",
    node_type=knext.NodeType.MANIPULATOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "LinearRegression.png",  # Placeholder
    id="linearregression",
    after="linearfit",
)
@knext.input_port(
    name="Training Image",
    description="GEE Image connection with multiple bands. The LAST band must be the dependent variable (Y). "
    "All other bands are independent variables (X₁, X₂, ..., Xₙ). "
    "Use **GEE Image Band Merger** to ensure correct band order. "
    "Constant term will be automatically added as the first band if missing.",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="Region",
    description="GEE Feature Collection connection with geometry for regression training area.",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with predicted values (if prediction enabled).",
    port_type=gee_image_port_type,
)
@knext.output_table(
    name="Regression Results",
    description="Table containing regression coefficients and RMSE.",
)
class LinearRegression:
    """Performs multiple linear regression using ee.Reducer.linearRegression.

    This node performs ordinary least squares (OLS) regression with multiple
    independent variables. The regression equation is: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε

    **Input Requirements:**

    - Training image bands (in order):
      1. Independent variables (X₁, X₂, ..., Xₙ)
      2. **Dependent variable (Y) - MUST BE THE LAST BAND**
    - Region: FeatureCollection or Geometry defining the training area
    - **Important**: Use **GEE Image Band Merger** node to ensure the dependent variable
      is placed as the last band before this node.

    **Features:**

    - **Automatic constant term**: A constant term (value 1) is automatically added
      as the first band if not present. This is required for estimating the intercept β₀.
    - **Automatic parameter inference**: numX is automatically calculated from the band count.
      numX = total_bands - 1 (including constant, excluding dependent variable).

    **Output:**

    - Regression coefficients: β₀ (constant), β₁, β₂, ... (one per independent variable)
    - Root Mean Square Error (RMSE)
    - Optional: Predicted image

    **Common Use Cases:**

    - Multi-band regression for tree cover estimation
    - Predicting continuous variables from multiple spectral bands
    - Building complex regression models with multiple predictors

    **Workflow Example:**

    1. Prepare independent variables (e.g., Landsat bands: B1, B2, B3, ...)
    2. Prepare dependent variable (e.g., Percent Tree Cover)
    3. Use **GEE Image Band Merger** to merge: [independent_vars, dependent_var]
    4. Connect merged image to this node (dependent variable must be last band)
    5. Node automatically adds constant term and performs regression

    **Note:** The constant term is automatically added as the first band.
    The dependent variable (Y) must be the last band in the training image.
    """

    use_nominal_scale, scale = knut.create_nominal_scale_parameters(
        max_value=1000,
        scale_description="Pixel resolution in meters for the regression calculation. Only used when Use NominalScale is disabled.",
    )

    best_effort = knext.BoolParameter(
        "Best effort",
        "If enabled, automatically adjusts scale to avoid maxPixels errors for large regions.",
        default_value=True,
    )

    max_pixels = knut.create_max_pixels_parameter(
        default_value=knut.GEE_MAX_PIXELS,
        description="Maximum number of pixels to use (only used if bestEffort is disabled).",
        is_advanced=True,
    ).rule(knext.OneOf(best_effort, [False]), knext.Effect.SHOW)

    generate_prediction = knext.BoolParameter(
        "Generate prediction image",
        "If enabled, applies the regression model to create a prediction image.",
        default_value=False,
    )

    predicted_band_name = knext.StringParameter(
        "Predicted band name",
        "Name for the output prediction band (only used if prediction is enabled).",
        default_value="predicted",
    ).rule(knext.OneOf(generate_prediction, [True]), knext.Effect.SHOW)

    def configure(self, configure_context, input_schema1, input_schema2):
        return None

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        training_image_connection,
        region_connection,
    ):
        import ee
        import logging
        import pandas as pd

        LOGGER = logging.getLogger(__name__)

        try:
            # Get image and region from connections
            training_image = training_image_connection.image
            region = region_connection.feature_collection

            # Get geometry from FeatureCollection
            if isinstance(region, ee.FeatureCollection):
                geometry = region.geometry()
            else:
                geometry = region

            # Get band names and count
            band_names = training_image.bandNames().getInfo()
            total_bands = len(band_names)

            if total_bands < 2:
                raise ValueError(
                    f"Training image must have at least 2 bands (independent variables + dependent variable). "
                    f"Found {total_bands} bands."
                )

            scale_value = knut.resolve_scale(
                self.use_nominal_scale, self.scale, training_image
            )

            # Automatically detect and add constant term if missing
            # Check if first band name suggests it's a constant term
            first_band_name = band_names[0].lower() if band_names else ""
            has_constant = first_band_name in [
                "constant",
                "const",
                "intercept",
                "bias",
            ] or first_band_name.startswith("constant")

            # Add constant term if missing
            if not has_constant:
                constant_band = ee.Image(1).rename("constant")
                # Get geometry for constant band to ensure proper coverage
                img_geometry = training_image.geometry()
                constant_band = constant_band.clip(img_geometry)
                # Add constant as first band
                training_image = constant_band.addBands(training_image)
                band_names = ["constant"] + band_names
                total_bands = len(band_names)
                LOGGER.warning(
                    "Constant term (value 1) automatically added as first band. "
                    "This is required for estimating the intercept β₀."
                )
            else:
                LOGGER.warning(
                    f"Using first band '{band_names[0]}' as constant term. "
                    "Ensure it has value 1 for all pixels."
                )

            # Auto-infer num_x from band count
            # Training image now has: [constant, X1, X2, ..., Xn, Y]
            # total_bands = 1 (constant) + num_independent + 1 (dependent variable)
            # The last band is always the dependent variable (num_y = 1)
            num_y_actual = 1  # Always assume single dependent variable (last band)
            num_x_actual = total_bands - num_y_actual  # Includes constant term

            if num_x_actual < 1:
                raise ValueError(
                    f"Cannot determine independent variables: total_bands ({total_bands}) - "
                    f"num_y (1) = {num_x_actual}. "
                    f"Need at least 2 bands total: independent variables + dependent variable. "
                    f"The dependent variable must be the LAST band."
                )

            LOGGER.warning(
                f"Auto-detected parameters: num_x = {num_x_actual} (including constant), "
                f"num_y = 1 (last band is dependent variable), total_bands = {total_bands}"
            )

            # Prepare reduceRegion parameters
            reduce_params = {
                "reducer": ee.Reducer.linearRegression(
                    numX=num_x_actual, numY=num_y_actual
                ),
                "geometry": geometry,
                "scale": scale_value,
            }

            if self.best_effort:
                reduce_params["bestEffort"] = True
            else:
                reduce_params["maxPixels"] = self.max_pixels

            # Perform linear regression
            regression_result = training_image.reduceRegion(**reduce_params)

            # Get coefficients
            coefficients_array = ee.Array(
                regression_result.get("coefficients")
            ).getInfo()

            # Extract coefficients (flatten if needed)
            if isinstance(coefficients_array, list) and len(coefficients_array) > 0:
                if isinstance(coefficients_array[0], list):
                    # 2D array, extract first column
                    coefficients = [row[0] for row in coefficients_array]
                else:
                    coefficients = coefficients_array
            else:
                coefficients = [coefficients_array]

            # Try to get RMSE from the result dictionary
            # According to GEE API, the key is 'residuals' (plural) and it's a matrix
            # The RMSE is the square root of the diagonal element [0][0]
            rmse = None
            try:
                # First check if 'residuals' key exists (plural - standard GEE API)
                regression_result_info = regression_result.getInfo()
                if "residuals" in regression_result_info:
                    residuals_array = ee.Array(
                        regression_result.get("residuals")
                    ).getInfo()
                    # Extract RMSE from residuals matrix (it's numY x numY, diagonal element)
                    if isinstance(residuals_array, list):
                        if isinstance(residuals_array[0], list):
                            # 2D matrix: get diagonal element [0][0] and take square root
                            if len(residuals_array) > 0 and len(residuals_array[0]) > 0:
                                residual_value = residuals_array[0][0]
                                if residual_value is not None:
                                    rmse = abs(float(residual_value)) ** 0.5
                        else:
                            # 1D array: take first element
                            if len(residuals_array) > 0:
                                residual_value = residuals_array[0]
                                if residual_value is not None:
                                    rmse = abs(float(residual_value)) ** 0.5
                elif "residual" in regression_result_info:
                    # Fallback to singular form (some older versions might use this)
                    residual_val = regression_result.get("residual").getInfo()
                    if isinstance(residual_val, (int, float)):
                        rmse = abs(float(residual_val)) ** 0.5
                    elif isinstance(residual_val, list) and len(residual_val) > 0:
                        residual_value = residual_val[0]
                        if residual_value is not None:
                            rmse = abs(float(residual_value)) ** 0.5
            except Exception as e:
                LOGGER.warning(f"Could not extract RMSE from regression result: {e}")
                rmse = None

            if rmse is None:
                LOGGER.warning(
                    "RMSE not available in regression result (possibly underdetermined system or insufficient samples)"
                )
                rmse = float("nan")

            # Log completion
            if rmse is not None and not (
                isinstance(rmse, float) and (rmse != rmse)
            ):  # Check if not NaN
                LOGGER.warning(
                    f"Linear regression completed: {len(coefficients)} coefficients, RMSE = {rmse:.6f}"
                )
            else:
                LOGGER.warning(
                    f"Linear regression completed: {len(coefficients)} coefficients, RMSE = N/A"
                )

            # Create results table
            # band_names already updated above
            num_coefs = len(coefficients)

            # Create coefficient names using actual band names
            coef_names = []
            for i in range(num_coefs):
                if i == 0:
                    # Constant term: use "constant"
                    coef_names.append("constant")
                elif i < num_x_actual:
                    # Independent variable coefficients: use actual band name
                    band_idx = i  # i=0 is constant, i=1 is first independent var
                    band_name = (
                        band_names[band_idx] if band_idx < len(band_names) else f"X{i}"
                    )
                    coef_names.append(band_name)
                else:
                    # Should not happen with proper num_x_actual, but handle gracefully
                    band_idx = i if i < len(band_names) else None
                    coef_names.append(band_names[band_idx] if band_idx else f"X{i}")

            # Create results table without description column
            results_df = pd.DataFrame(
                {
                    "coefficient": coef_names[:num_coefs],
                    "value": coefficients[:num_coefs],
                }
            )

            # Add RMSE row (only if RMSE is available)
            if rmse is not None and not (
                isinstance(rmse, float) and (rmse != rmse)
            ):  # Check if not NaN
                rmse_row = pd.DataFrame(
                    {
                        "coefficient": ["RMSE"],
                        "value": [rmse],
                    }
                )
                results_df = pd.concat([results_df, rmse_row], ignore_index=True)
            else:
                LOGGER.warning("Skipping RMSE row in results table (not available)")

            # Generate prediction image if requested
            output_image = training_image
            if self.generate_prediction:
                # Always use first num_x_actual bands (constant + independent variables)
                # Training image: [constant, X1, X2, ..., Xn, Y]
                # Prediction needs: [constant, X1, X2, ..., Xn] = first num_x_actual bands
                # Following the same logic as training: last band is dependent variable,
                # all previous bands are independent variables (including constant)
                pred_image = training_image.select(band_names[:num_x_actual])

                # Verify band count matches coefficients count
                # Expected: num_x_actual bands (constant + independent variables)
                # Coefficients: num_x_actual (beta0, beta1, ..., beta_{num_x_actual-1})
                actual_num_bands = pred_image.bandNames().size().getInfo()
                expected_num_bands = num_x_actual

                if actual_num_bands != expected_num_bands:
                    raise ValueError(
                        f"Band count mismatch: prediction image has {actual_num_bands} bands, "
                        f"but regression has {expected_num_bands} coefficients. "
                        f"Expected: first {expected_num_bands} bands from training image "
                        f"(constant + {num_x_actual - 1} independent variables, excluding dependent variable)"
                    )

                # Apply regression: sum of (coefficient * band); coefficients in order: [beta0 (constant), beta1, beta2, ..., beta_{num_x_actual-1}]
                coeff_image = ee.Image.constant(coefficients[:expected_num_bands])
                predicted = (
                    pred_image.multiply(coeff_image)
                    .reduce(ee.Reducer.sum())
                    .rename(self.predicted_band_name)
                )

                output_image = predicted
                LOGGER.warning(
                    f"Prediction image generated with band name '{self.predicted_band_name}': "
                    f"{actual_num_bands} bands used, {expected_num_bands} coefficients"
                )

            return (
                knut.export_gee_image_connection(
                    output_image, training_image_connection
                ),
                knext.Table.from_pandas(results_df),
            )

        except Exception as e:
            LOGGER.error(f"Linear regression failed: {e}")
            raise


############################################
# CART Regression
############################################


@knext.node(
    name="GEE CART Regression Learner",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "CARTRegression.png",
    id="cartregressionlearner",
    after="linearregression",
)
@knext.input_port(
    name="Training Data",
    description="GEE Feature Collection connection with pixel values and continuous numeric labels (not categorical).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained CART regression model.",
    port_type=gee_classifier_port_type,
)
class CARTRegressionLearner:
    """Trains a CART model in regression mode for continuous targets.

    This node extends the existing CART classifier to support regression mode,
    allowing prediction of continuous numeric values instead of categorical classes.

    **Input Requirements:**

    - Training FeatureCollection with:
      - Feature properties: Independent variables (band values)
      - Target property: Continuous numeric dependent variable (e.g., tree cover percentage)

    **Output:**

    - Trained CART regression model (can be used with **Image Regression Predictor** node)

    **Parameters:**

    - **Max Nodes**: Maximum number of nodes in the decision tree
    - **Min Leaf Population**: Minimum samples required in a leaf node

    **Common Use Cases:**

    - Non-linear regression for continuous value prediction
    - Tree-based regression models
    - Handling non-linear relationships between variables

    **Workflow:**

    1. Sample training data: `Training Image` → **Image Cluster Sampling** → `FeatureCollection`
    2. Train model: `FeatureCollection` → **CART Regression Learner** → `Classifier Connection`
    3. Predict: `Prediction Image` + `Classifier Connection` → **Image Regression Predictor** → `Predicted Image`

    **Note:** This is similar to **GEE CART Learner** but outputs a regression model.
    The trained model must be used with **Image Regression Predictor** node (not **Image Class Predictor**).
    """

    target_property = knext.StringParameter(
        "Target property",
        "Property name containing continuous numeric target (dependent variable) in the training FeatureCollection (e.g., 'Percent_Tree_Cover', 'biomass').",
        default_value="label",
    )

    bands = knext.StringParameter(
        "Bands/Features",
        """Comma-separated list of band/feature names to use as independent variables (X) for training (e.g., 'B2,B3,B4,B8').
        Leave empty to use all properties except target.""",
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
                    "Input must be a FeatureCollection from sampling nodes"
                )

            # Get available properties
            try:
                sample_feature = training_fc.first().getInfo()
                available_properties = list(sample_feature.get("properties", {}).keys())
                system_props = ["system:index", "system:time_start"]
                available_features = [
                    p
                    for p in available_properties
                    if p not in system_props and p != self.target_property
                ]
                LOGGER.warning(
                    f"Available properties: {available_features} (excluding system properties and target)"
                )
            except Exception as e:
                LOGGER.warning(f"Could not inspect FeatureCollection properties: {e}")
                available_features = []

            # Determine feature list (bands to use)
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
                    f"No valid features found. Available properties: {available_properties}. "
                    f"Target property: {self.target_property}. "
                    f"Please check 'Bands/Features' parameter."
                )

            LOGGER.warning(
                f"Using {len(feature_list)} features for training: {feature_list[:10]}{'...' if len(feature_list) > 10 else ''}"
            )

            # Build classifier parameters
            classifier_params = {
                "maxNodes": self.max_nodes,
                "minLeafPopulation": self.min_leaf_population,
            }

            # Create and train CART classifier in REGRESSION mode
            classifier = ee.Classifier.smileCart(**classifier_params).setOutputMode(
                "REGRESSION"
            )

            trained_classifier = classifier.train(
                features=training_fc,
                classProperty=self.target_property,
                inputProperties=feature_list,
            )

            LOGGER.warning(
                f"Successfully trained CART regression model with max {self.max_nodes} nodes"
            )

            # Create classifier connection object
            classifier_connection = knut.export_gee_classifier_connection(
                classifier=trained_classifier,
                existing_connection=training_data_connection,
                training_data=training_fc,
                label_property=self.target_property,  # Internal API uses label_property, but value is target
                reverse_mapping=None,  # Not needed for regression
                input_properties=feature_list,
            )

            return classifier_connection

        except Exception as e:
            LOGGER.error(f"CART regression training failed: {e}")
            raise


############################################
# Image Regression Predictor
############################################


@knext.node(
    name="GEE Image Regression Predictor",
    node_type=knext.NodeType.PREDICTOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "ImageRegressPredictor.png",
    id="imageregressionpredictor",
    after="cartregressionlearner",
)
@knext.input_port(
    name="GEE Image Connection",
    description="GEE Image connection with image to predict (must contain independent variable bands).",
    port_type=gee_image_port_type,
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained regression model (from CART Regression Learner).",
    port_type=gee_classifier_port_type,
)
@knext.output_port(
    name="GEE Image Connection",
    description="GEE Image connection with predicted continuous values.",
    port_type=gee_image_port_type,
)
class ImageRegressionPredictor:
    """Applies a trained regression model to predict continuous values in an image.

    This node uses a trained regression model (CART Regression) to predict
    continuous numeric values for each pixel in an image.

    **Input Requirements:**

    - **Image**: Multi-band image (must contain the independent variable bands used during training)
    - **Classifier**: Trained regression model from **CART Regression Learner** node

    **Automatic Band Selection:**

    - The node automatically uses the same bands that were used during training
    - Bands are inherited from the classifier connection (no manual selection needed)
    - The input image must contain all the independent variable bands used during training

    **Output:**

    The node outputs an image with one band:
    - **Predicted band**: Continuous numeric values (configurable band name, default: 'predicted')

    **Key Differences from Image Class Predictor:**

    - Outputs continuous numeric values (not categorical class labels)
    - Output band name is configurable (not fixed to 'GEE_class')
    - No class value remapping (regression outputs are already continuous values)
    - Only works with regression models (use **Image Class Predictor** for classification)

    **Common Use Cases:**

    - Predicting tree cover percentage
    - Estimating biomass or crop yield
    - Continuous value mapping (temperature, precipitation, etc.)
    - Non-linear regression prediction

    **Workflow:**

    Training Image (all bands: X + Y) → **Image Cluster Sampling** → FeatureCollection →
    **CART Regression Learner** → Classifier Connection →
    **Image Regression Predictor** (with Prediction Image containing X bands only) → Predicted Image
    """

    predicted_band_name = knext.StringParameter(
        "Predicted band name",
        "Name for the output prediction band (e.g., 'predicted', 'cartRegression', 'treeCover').",
        default_value="predicted",
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
            image = image_connection.image
            classifier = classifier_connection.classifier

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please use a trained regression model from CART Regression Learner node."
                )

            # Get input properties (bands) from classifier connection
            input_properties = classifier_connection.input_properties
            if input_properties:
                # Select only the bands used during training
                image = image.select(input_properties)
                LOGGER.warning(
                    f"Using training bands for regression prediction: {input_properties}"
                )
            else:
                LOGGER.warning(
                    "No input properties found in classifier connection, using all bands"
                )

            # Apply regression model - get prediction (continuous values)
            # Output band name is configurable (not fixed like classification)
            predicted_image = image.classify(
                classifier, outputName=self.predicted_band_name
            )

            # No reverse_mapping needed for regression (outputs are already continuous)
            # Regression models output continuous numeric values directly

            LOGGER.warning(
                f"Successfully predicted continuous values with band name '{self.predicted_band_name}'"
            )

            return knut.export_gee_image_connection(predicted_image, image_connection)

        except Exception as e:
            LOGGER.error(f"Image regression prediction failed: {e}")
            raise


############################################
# Feature Collection Regression Predictor
############################################


@knext.node(
    name="GEE Feature Collection Regression Predictor",
    node_type=knext.NodeType.PREDICTOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "FCRegression.png",
    id="featurecollectionregressionpredictor",
    after="imageregressionpredictor",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with features to predict (must contain the same band/property names used for training).",
    port_type=gee_feature_collection_port_type,
)
@knext.input_port(
    name="GEE Classifier Connection",
    description="GEE Classifier connection with trained regression model (from CART Regression Learner).",
    port_type=gee_classifier_port_type,
)
@knext.output_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with prediction results added as a new property.",
    port_type=gee_feature_collection_port_type,
)
class FeatureCollectionRegressionPredictor:
    """Applies a trained regression model to predict continuous values for features in a Feature Collection.

    This node uses a trained regression model (CART Regression) to predict continuous numeric values
    for each feature in a Feature Collection. The prediction is added as a new property to each feature,
    allowing you to export and validate results.

    **Input Requirements:**

    - **Feature Collection**: Must contain the same band/property names used during training
      (e.g., 'B2', 'B3', 'B4' if those were the training bands)
    - **Classifier**: Trained regression model from **CART Regression Learner** node

    **Output:**

    - Original Feature Collection with a new prediction property containing predicted continuous values
    - All original properties are preserved (including target property if present)
    - Can be exported to table for validation

    **Common Use Cases:**

    - **Validation**: Use training data FeatureCollection, then use this node to predict values
      and compare with actual target values using **Regression Scorer**
    - **Independent Testing**: Apply trained regression model to independent test datasets
    - **Cross-Validation**: Predict on held-out validation sets
    - **Spatial Validation**: Predict on features from different regions or time periods

    **Workflow Example:**

    1. Sample training data: `Training Image` → **Image Cluster Sampling** → `FeatureCollection`
    2. Train model: `FeatureCollection` → **CART Regression Learner** → `Classifier Connection`
    3. Predict: `FeatureCollection` + `Classifier Connection` → **Feature Collection Regression Predictor** (this node)
    4. Score: Predicted FeatureCollection → **Regression Scorer** → Metrics

    **Band/Property Matching:**

    - The Feature Collection must contain properties with the same names as the bands/features
      used during training
    - If training used 'B2', 'B3', 'B4', the Feature Collection must have these properties
    - System properties (e.g., 'system:index') are automatically excluded

    **Output Property:**

    The prediction results are stored in a property named **'predicted'** (configurable, default: 'predicted').

    """

    predicted_property_name = knext.StringParameter(
        "Predicted property name",
        "Name for the output prediction property (e.g., 'predicted', 'cartRegression', 'treeCover').",
        default_value="predicted",
    )

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
                    "Input must be a FeatureCollection (e.g., from 'Image Cluster Sampling')"
                )

            if classifier is None:
                raise ValueError(
                    "Classifier connection does not contain a trained classifier. "
                    "Please use a trained regression model from CART Regression Learner node."
                )

            # Classify (predict) the Feature Collection
            # For regression, this outputs continuous values
            predicted_fc = feature_collection.classify(
                classifier, outputName=self.predicted_property_name
            )

            # No reverse_mapping needed for regression (outputs are already continuous)
            # Regression models output continuous numeric values directly

            LOGGER.warning(
                f"Successfully predicted continuous values for Feature Collection with property '{self.predicted_property_name}'"
            )

            # Log sample count if available
            try:
                fc_size = feature_collection.size().getInfo()
                LOGGER.warning(f"Predicted values for {fc_size} features")
            except Exception:
                LOGGER.warning("Predicted Feature Collection (size check skipped)")

            return knut.export_gee_feature_collection_connection(
                predicted_fc, fc_connection
            )

        except Exception as e:
            LOGGER.error(f"Feature Collection regression prediction failed: {e}")
            raise


############################################
# Regression Scorer
############################################


@knext.node(
    name="GEE Regression Scorer",
    node_type=knext.NodeType.OTHER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "RegressionScorer.png",
    id="regressionscorer",
    after="featurecollectionregressionpredictor",
)
@knext.input_port(
    name="GEE Feature Collection Connection",
    description="GEE Feature Collection connection with predicted features (from Feature Collection Regression Predictor). Must contain both target property (actual values) and prediction property (predicted values).",
    port_type=gee_feature_collection_port_type,
)
@knext.output_table(
    name="Regression Metrics",
    description="Table containing RMSE, R², MAE, Correlation, and other regression metrics.",
)
class RegressionScorer:
    """Computes regression performance metrics from predicted FeatureCollection.

    This node evaluates regression model performance by comparing actual (target) values
    with predicted values in a FeatureCollection that has already been predicted by the
    **Feature Collection Regression Predictor** node. It calculates performance metrics
    including RMSE, R², MAE, and correlation using GEE's server-side reducers.

    **Input Requirements:**

    - **Predicted Feature Collection**: FeatureCollection output from **Feature Collection Regression Predictor** node
    - Must contain:
      - **Target property** (e.g., 'Percent_Tree_Cover'): Actual/observed continuous values
      - **Prediction property** (e.g., 'predicted'): Predicted continuous values (automatically added by Feature Collection Regression Predictor)

    **Output Metrics:**

    - **RMSE**: Root Mean Square Error
    - **R²**: Coefficient of determination (R-squared)
    - **MAE**: Mean Absolute Error
    - **Correlation**: Pearson correlation coefficient
    - **Sample Size**: Number of samples used for evaluation

    **Workflow:**

    1. Sample training data: `Training Image` → **Image Cluster Sampling** → `FeatureCollection`
    2. Train model: `FeatureCollection` → **CART Regression Learner** → `Classifier Connection`
    3. Predict: `FeatureCollection` + `Classifier Connection` → **Feature Collection Regression Predictor** → Predicted FeatureCollection
    4. Score: Predicted FeatureCollection → **Regression Scorer** (this node) → Metrics

    **Performance:**

    - Uses GEE's reducers for efficient server-side computation
    - All metrics computed server-side, minimizing data transfer
    - Only final metrics are downloaded

    **Common Use Cases:**

    - Evaluate regression model performance on training/validation data
    - Compare model performance across different configurations
    - Validate regression model accuracy
    """

    target_property = knext.StringParameter(
        "Target property",
        "Name of the property containing actual/observed continuous values (e.g., 'Percent_Tree_Cover', 'biomass'). This should match the target property used during training.",
        default_value="label",
    )

    predicted_property = knext.StringParameter(
        "Predicted property",
        "Name of the property containing predicted continuous values (e.g., 'predicted', 'cartRegression'). This should match the output property name from Feature Collection Regression Predictor.",
        default_value="predicted",
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
            # Get predicted FeatureCollection from connection
            predicted_fc = fc_connection.feature_collection

            if not isinstance(predicted_fc, ee.FeatureCollection):
                raise ValueError(
                    "Input must be a FeatureCollection. "
                    "Please connect the output from 'Feature Collection Regression Predictor' node."
                )

            # Verify that the FeatureCollection contains the required properties
            try:
                first_feature = predicted_fc.first()
                properties = first_feature.propertyNames().getInfo()
                if self.predicted_property not in properties:
                    raise ValueError(
                        f"FeatureCollection does not contain prediction property '{self.predicted_property}'. "
                        f"Please ensure the FeatureCollection has been predicted by 'Feature Collection Regression Predictor' node. "
                        f"Found properties: {properties}"
                    )
                if self.target_property not in properties:
                    raise ValueError(
                        f"FeatureCollection does not contain target property '{self.target_property}'. "
                        f"Please check the target property name. "
                        f"Found properties: {properties}"
                    )
            except Exception as e:
                LOGGER.warning(f"Could not verify properties: {e}")

            # Get data size for logging
            try:
                fc_size = predicted_fc.size().getInfo()
                LOGGER.warning(
                    f"Computing regression metrics for {fc_size} features "
                    f"(target: {self.target_property}, predicted: {self.predicted_property})"
                )
            except Exception:
                LOGGER.warning(
                    f"Computing regression metrics (target: {self.target_property}, predicted: {self.predicted_property})"
                )

            # Calculate metrics using GEE reducers
            # Use reduceColumns to compute all metrics in one operation
            def compute_metrics(fc, actual_prop, pred_prop):
                """Compute regression metrics using GEE reducers."""

                # Calculate residuals for each feature
                def add_residual(feature):
                    actual = ee.Number(feature.get(actual_prop))
                    predicted = ee.Number(feature.get(pred_prop))
                    residual = actual.subtract(predicted)
                    residual_sq = residual.pow(2)
                    residual_abs = residual.abs()
                    return (
                        feature.set("residual", residual)
                        .set("residual_sq", residual_sq)
                        .set("residual_abs", residual_abs)
                        .set("actual", actual)
                        .set("predicted", predicted)
                    )

                # Add residual properties
                fc_with_residuals = fc.map(add_residual)

                # Calculate means separately to avoid duplicate output names
                # Use aggregate_array and then reduce to calculate means
                actual_list = fc_with_residuals.aggregate_array(actual_prop)
                predicted_list = fc_with_residuals.aggregate_array(pred_prop)
                residual_sq_list = fc_with_residuals.aggregate_array("residual_sq")
                residual_abs_list = fc_with_residuals.aggregate_array("residual_abs")

                # Calculate means using reducers
                actual_mean = actual_list.reduce(ee.Reducer.mean())
                predicted_mean = predicted_list.reduce(ee.Reducer.mean())
                mean_residual_sq = residual_sq_list.reduce(ee.Reducer.mean())
                mean_residual_abs = residual_abs_list.reduce(ee.Reducer.mean())

                # Calculate RMSE = sqrt(mean(residuals^2))
                rmse = ee.Number(mean_residual_sq).sqrt()

                # Calculate MAE = mean(|residuals|)
                mae = ee.Number(mean_residual_abs)

                # Calculate correlation and R² using reduceColumns
                # We need to compute:
                # - numerator = sum((actual - actual_mean) * (predicted - predicted_mean))
                # - actual_var = sum((actual - actual_mean)^2)
                # - predicted_var = sum((predicted - predicted_mean)^2)
                # - ss_res = sum((actual - predicted)^2) = sum(residual_sq)

                def add_centered_values(feature):
                    actual = ee.Number(feature.get(actual_prop))
                    predicted = ee.Number(feature.get(pred_prop))
                    actual_centered = actual.subtract(ee.Number(actual_mean))
                    predicted_centered = predicted.subtract(ee.Number(predicted_mean))
                    # Also get residual_sq from fc_with_residuals
                    residual_sq = ee.Number(feature.get("residual_sq"))
                    return (
                        feature.set("actual_centered", actual_centered)
                        .set("predicted_centered", predicted_centered)
                        .set("actual_centered_sq", actual_centered.pow(2))
                        .set("predicted_centered_sq", predicted_centered.pow(2))
                        .set(
                            "cross_product",
                            actual_centered.multiply(predicted_centered),
                        )
                        .set("residual_sq", residual_sq)  # Preserve residual_sq
                    )

                # Use fc_with_residuals (which has residual_sq) instead of fc
                fc_centered = fc_with_residuals.map(add_centered_values)

                # Calculate sums separately to avoid duplicate output names
                # Use aggregate_array and then reduce to calculate sums
                actual_centered_sq_list = fc_centered.aggregate_array(
                    "actual_centered_sq"
                )
                predicted_centered_sq_list = fc_centered.aggregate_array(
                    "predicted_centered_sq"
                )
                cross_product_list = fc_centered.aggregate_array("cross_product")
                residual_sq_list = fc_centered.aggregate_array("residual_sq")

                # Calculate sums using reducers
                actual_var = actual_centered_sq_list.reduce(ee.Reducer.sum())
                predicted_var = predicted_centered_sq_list.reduce(ee.Reducer.sum())
                numerator = cross_product_list.reduce(ee.Reducer.sum())
                ss_res = residual_sq_list.reduce(ee.Reducer.sum())

                # Correlation = numerator / sqrt(actual_var * predicted_var)
                variance_product = ee.Number(actual_var).multiply(
                    ee.Number(predicted_var)
                )
                correlation_raw = ee.Number(numerator).divide(
                    ee.Number(variance_product)
                    .sqrt()
                    .add(1e-10)  # Add epsilon to avoid division by zero
                )
                # Clamp correlation to valid range [-1, 1]
                correlation = ee.Number(correlation_raw).max(-1).min(1)

                # R² = 1 - (SS_res / SS_tot)
                # SS_tot = actual_var
                ss_tot = ee.Number(actual_var)
                ss_res_num = ee.Number(ss_res)
                r_squared = ee.Number(1).subtract(
                    ss_res_num.divide(
                        ss_tot.add(1e-10)
                    )  # Add epsilon to avoid division by zero
                )

                # Sample size
                sample_size = fc.size()

                # Return as dictionary
                return ee.Dictionary(
                    {
                        "RMSE": rmse,
                        "R_squared": r_squared,
                        "MAE": mae,
                        "Correlation": correlation,
                        "Sample_Size": sample_size,
                    }
                )

            # Compute metrics
            metrics_dict = compute_metrics(
                predicted_fc, self.target_property, self.predicted_property
            )
            metrics_info = metrics_dict.getInfo()

            # Create results DataFrame
            metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": "RMSE",
                        "Value": round(float(metrics_info.get("RMSE", 0.0)), 6),
                    },
                    {
                        "Metric": "R²",
                        "Value": round(float(metrics_info.get("R_squared", 0.0)), 6),
                    },
                    {
                        "Metric": "MAE",
                        "Value": round(float(metrics_info.get("MAE", 0.0)), 6),
                    },
                    {
                        "Metric": "Correlation",
                        "Value": round(float(metrics_info.get("Correlation", 0.0)), 6),
                    },
                    {
                        "Metric": "Sample Size",
                        "Value": int(metrics_info.get("Sample_Size", 0)),
                    },
                ]
            )

            LOGGER.warning(
                f"Successfully computed regression metrics: RMSE={metrics_info.get('RMSE', 0):.6f}, "
                f"R²={metrics_info.get('R_squared', 0):.6f}, MAE={metrics_info.get('MAE', 0):.6f}"
            )

            return knext.Table.from_pandas(metrics_df)

        except Exception as e:
            LOGGER.error(f"Regression scoring failed: {e}")
            raise
