CREATE OR REPLACE VIEW view_localworker2_129600423_0_0 AS
SELECT "row_id",
       "rightttgtransversetemporalgyrus",
       "leftpinsposteriorinsula",
       "leftpoparietaloperculum",
       "rightptplanumtemporale",
       "leftventraldc"
FROM "dementia:0.1"."primary_data"
WHERE ("rightttgtransversetemporalgyrus" IS NOT NULL
   AND "leftpinsposteriorinsula"       IS NOT NULL
   AND "leftpoparietaloperculum"       IS NOT NULL
   AND "rightptplanumtemporale"        IS NOT NULL
   AND "leftventraldc"                 IS NOT NULL
   AND "alzheimerbroadcategory"        IS NOT NULL
   AND "dataset" IN ('desd-synthdata0','ppmi0'));

CREATE OR REPLACE VIEW view_localworker2_129600423_0_1 AS
SELECT "row_id",
       "alzheimerbroadcategory"
FROM "dementia:0.1"."primary_data"
WHERE ("rightttgtransversetemporalgyrus" IS NOT NULL
   AND "leftpinsposteriorinsula"       IS NOT NULL
   AND "leftpoparietaloperculum"       IS NOT NULL
   AND "rightptplanumtemporale"        IS NOT NULL
   AND "leftventraldc"                 IS NOT NULL
   AND "alzheimerbroadcategory"        IS NOT NULL
   AND "dataset" IN ('desd-synthdata0','ppmi0'));

CREATE TABLE IF NOT EXISTS normal_localworker2_129600423_1_0
(
    "row_id"                          INT,
    "intercept"                       DOUBLE,
    "rightttgtransversetemporalgyrus" DOUBLE,
    "leftpinsposteriorinsula"         DOUBLE,
    "leftpoparietaloperculum"         DOUBLE,
    "rightptplanumtemporale"          DOUBLE,
    "leftventraldc"                   DOUBLE
);

INSERT INTO normal_localworker2_129600423_1_0
SELECT "row_id",
       1 AS "intercept",
       "rightttgtransversetemporalgyrus",
       "leftpinsposteriorinsula",
       "leftpoparietaloperculum",
       "rightptplanumtemporale",
       "leftventraldc"
FROM view_localworker2_129600423_0_0
WHERE NOT EXISTS (SELECT * FROM normal_localworker2_129600423_1_0);

CREATE TABLE IF NOT EXISTS normal_localworker2_129600423_2_0
(
    "row_id" INT,
    "ybin"   INT
);

CREATE OR REPLACE FUNCTION LabelBinarizer___transform_local_rgho_2_129600423
(
    "y_row_id"                INT,
    "y_alzheimerbroadcategory" VARCHAR(500)
)
RETURNS TABLE("row_id" INT, "ybin" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    y = udfio.from_relational_table(
            {name: _columns[name_w_prefix]
             for name, name_w_prefix in zip(
                 ['row_id', 'alzheimerbroadcategory'],
                 ['y_row_id', 'y_alzheimerbroadcategory'])},
            'row_id')
    positive_class = 'Other'
    ybin = y == positive_class
    result = pd.DataFrame({'ybin': ybin.to_numpy().reshape((-1,))},
                          index=y.index)
    return udfio.as_relational_table(result, 'row_id')
};

SELECT
    *
FROM
    LabelBinarizer___transform_local_rgho_2_813997181((
        SELECT
            view_localworker2_813997181_0_1."row_id",
            view_localworker2_813997181_0_1."alzheimerbroadcategory"
        FROM
            view_localworker2_813997181_0_1
    ))
WHERE NOT EXISTS (SELECT * FROM normal_localworker2_813997181_2_0);
