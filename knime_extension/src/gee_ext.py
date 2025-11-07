# The root category of all gee categories
import knime_extension as knext

# This defines the root gee KNIME category that is displayed in the node repository
category = knext.category(
    path="/community",
    level_id="gee",  # this is the id of the category in the node repository #FIXME:
    name="Google Earth Engine",
    description="Nodes for Google Earth Engine",
    # starting at the root folder of the extension_module parameter in the knime.yml file
    icon="icons/GEE.png",
)


# need import node files here
import nodes.authorization
import nodes.image
import nodes.imagecollection
import nodes.featureio
import nodes.visualize
import nodes.sampling
import nodes.supervise

# import nodes.tool
# import nodes.cluster
# import nodes.classifier
# import nodes.io
# import nodes.manipulator
# import nodes.dataset
