# The root category of all gee categories
import knime_extension as knext

# This defines the root gee KNIME category that is displayed in the node repository
category = knext.category(
    path="/community",
    level_id="gee", # this is the id of the category in the node repository #FIXME: 
    name="gee",
    description="KNIME Extension for Google Earth Engine",
    # starting at the root folder of the extension_module parameter in the knime.yml file
    icon="icons/knime-gee.png",
)


# need import node files here
# import nodes.authorization
import nodes.io
import nodes.manipulator
import nodes.classifier
import nodes.cluster
import nodes.visualize
import nodes.dataset
