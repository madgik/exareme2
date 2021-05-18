from mipengine.common.common_data_elements import CommonDataElement
from mipengine.common.common_data_elements import CommonDataElements
from mipengine.common.common_data_elements import MetadataEnumeration
from mipengine.common.common_data_elements import MetadataVariable

common_data_elements = CommonDataElements()
common_data_elements.pathologies = {
    "test_pathology1": {
        "test_cde1": CommonDataElement(
            MetadataVariable(
                code="test_cde1",
                label="test cde1",
                sql_type="int",
                isCategorical=False,
                enumerations=None,
                min=None,
                max=None,
            )
        ),
        "test_cde2": CommonDataElement(
            MetadataVariable(
                code="test_cde2",
                label="test cde2",
                sql_type="real",
                isCategorical=False,
                enumerations=None,
                min=None,
                max=None,
            )
        ),
        "test_cde3": CommonDataElement(
            MetadataVariable(
                code="test_cde3",
                label="test cde3",
                sql_type="int",
                isCategorical=True,
                enumerations=[
                    MetadataEnumeration(code="1", label="1"),
                    MetadataEnumeration(code="2", label="2"),
                ],
                min=None,
                max=None,
            )
        ),
    }
}
