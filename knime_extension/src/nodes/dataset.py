import knime_extension as knext
import util.knime_utils as knut


__category = knext.category(
    path="/community/gee",
    level_id="dataset",
    name="Dataset",
    description="Dataset nodes for Google Earth Engine",
    icon="icons/dataset.png",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/dataset/"


############################################
# GEE Harmonized Sentinel-2 MSI Data node
############################################
@knext.node(
    name="GEE Harmonized Sentinel-2 MSI Data",
    node_type=knext.NodeType.SOURCE,
    category=__category,
    icon_path=__NODE_ICON_PATH + "dataset.png",
    after="",
)
# @knext.input_binary(
#     name="Region",
#     description="The region to constrain the data to.",
#     id="geemap.gee.Image",
# )
@knext.output_binary(
    name="Harmonized Sentinel-2 MSI Data",
    description="The output binary containing the GEE Harmonized Sentinel-2 MSI Data.",
    id="geemap.gee.ImageCollection",
)
@knext.output_view(
    name="ImageCollection Info view",
    description="Showing a map with the GEE map",
)
class GEEHarmonizedSentinel2MSIData:
    """GEE Harmonized Sentinel-2 MSI Data.
    GEE Harmonized Sentinel-2 MSI Data node.
    """

    start_date = knext.StringParameter(
        "Start Date",
        "The start date for the GEE Harmonized Sentinel-2 MSI Data",
        default_value="2024-01-01",
    )

    end_date = knext.StringParameter(
        "End Date",
        "The end date for the GEE Harmonized Sentinel-2 MSI Data",
        default_value="2024-03-20",
    )

    # TODO: add region to constrain the data to a specific region
    # Function to mask clouds using the Sentinel-2 QA band
    def mask_s2_clouds(self, image):
        import ee

        ee.Authenticate()
        ee.Initialize()
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    def configure(self, configure_context):
        return None

    def execute(self, exec_context: knext.ExecutionContext):
        import ee

        ee.Authenticate()
        ee.Initialize()
        import pickle

        # feature_collection = pickle.loads(input_binary)
        harmonized_sentinel2_msi_data = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(self.mask_s2_clouds)
        )

        info = harmonized_sentinel2_msi_data.first().getInfo()
        import json

        json_string = json.dumps(info, indent=4)
        html = f"""<h2>ImageCollection Info</h3><pre>{json_string}</pre>"""
        harmonized_sentinel2_msi_data_string = pickle.dumps(
            harmonized_sentinel2_msi_data
        )
        return harmonized_sentinel2_msi_data_string, knext.view_html(html)
