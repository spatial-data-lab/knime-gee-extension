import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="classifier",
    name="Classifier",
    description="Classifier nodes for Google Earth Engine",
    icon="icons/classifier.png",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/classifier/"


############################################
# SmileCart Classifier node
############################################
@knext.node(
    name="SmileCart Classifier",
    node_type=knext.NodeType.LEARNER,
    category=__category,
    icon_path=__NODE_ICON_PATH + "classifier.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.input_binary(
    name="Feature Collection",
    description="The input binary containing the GEE Feature Collection.",
    id="geemap.gee.Image",
)
@knext.output_binary(
    name="The classifier",
    description="The output binary of the classifier.",
    id="geemap.gee.classifier",
)
class SmileCartClassifier:
    """SmileCart Classifier.
    SmileCart Classifier node.
    """

    bands = knext.StringParameter(
        "Bands",
        "The bands to use for the classifier",
        default_value="B2,B3,B4,B8",
    )

    class_property = knext.StringParameter(
        "Class Property",
        "The class property to use for the classifier",
        default_value="LC",
    )

    scale = knext.IntParameter(
        "Scale",
        "The scale to use for the classifier",
        default_value=30,
    )

    def configure(self, configure_context, input_schema, input_schema_1):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, input_binary, input_binary_1
    ):
        import ee
        ee.Authenticate()
        ee.Initialize()
        import pickle

        image = pickle.loads(input_binary)
        feature_collection = pickle.loads(input_binary_1)

        # training_sample = table.filterBounds(geometry)
        # print(training_sample.getInfo())
        # Bands typically used for Sentinel-2 imagery analysis
        # import json
        bands = [b.strip(" ") for b in self.bands.split(",")]
        # bands = ['B2', 'B3', 'B4', 'B8']

        training = image.select(bands).sampleRegions(
            collection=feature_collection,
            properties=[self.class_property],
            scale=self.scale,
        )

        classifier = ee.Classifier.smileCart().train(
            features=training, classProperty=self.class_property, inputProperties=bands
        )

        # classifier = ee.Classifier.smileCart().train(feature_collection, 'class', image.bandNames())

        classifier_string = pickle.dumps(classifier)
        return classifier_string


############################################
# SimpleCart Predictor node
############################################
@knext.node(
    name="SimpleCart Predictor",
    node_type=knext.NodeType.PREDICTOR,
    category=__category,
    icon_path=__NODE_ICON_PATH + "classifier.png",
    after="",
)
@knext.input_binary(
    name="Image",
    description="The input binary containing the GEE Image.",
    id="geemap.gee.Image",
)
@knext.input_binary(
    name="Classifier",
    description="The input binary of the classifier.",
    id="geemap.gee.classifier",
)
@knext.output_binary(
    name="Predicted Image",
    description="The output binary containing the predicted image.",
    id="geemap.gee.Image",
)
class SimpleCartPredictor:
    """SimpleCart Predictor.
    SimpleCart Predictor node.
    """

    bands = knext.StringParameter(
        "Bands",
        "The bands to use for the classifier",
        default_value="B2,B3,B4,B8",
    )

    def configure(self, configure_context, input_schema, input_schema_1):
        return None

    def execute(
        self, exec_context: knext.ExecutionContext, input_binary, input_binary_1
    ):
        import ee
        ee.Authenticate()
        ee.Initialize()
        import pickle

        image = pickle.loads(input_binary)
        classifier = pickle.loads(input_binary_1)
        bands = [b.strip(" ") for b in self.bands.split(",")]
        predicted_image = image.select(bands).classify(classifier)

        predicted_image_string = pickle.dumps(predicted_image)
        return predicted_image_string
