from mipengine.common_data_elements import CommonDataElements
from mipengine.controller import config as controller_config


def get_cdes():
    controller_common_data_elements = CommonDataElements(
        controller_config.cdes_metadata_path
    )
    return controller_common_data_elements.data_models
