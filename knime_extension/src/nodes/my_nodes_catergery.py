import knime_extension as knext
import util.knime_utils as knut



__category = knext.category(
    path="/community/gee",
    level_id="my_nodes_catergery",
    name="My first category",
    description="Nodes for my first category",
    # starting at the root folder of the extension_module parameter in the knime.yml file
    icon="icons/icon/CategoryIcon.png",
    after="some other category",
)

# Root path for all node icons in this file
__NODE_ICON_PATH = "icons/icon/OpenDataset/my_first_category"
