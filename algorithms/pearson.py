from db_data import Data


class Algorithm:
    async def run(self, data: Data) -> list:

        data2 = await data.run_udf("local_pearson")

        #data4 = await data.run_udf("local_udf")

        data3 = await data2.run_udf("global_pearson")

        #data5 = await data.run_udf("global_udf")

        #result1 = await data5.get_value()

        #result2 = await data3.get_value()

        return await data3.get_value()
