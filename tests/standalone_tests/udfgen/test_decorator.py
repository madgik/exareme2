# type: ignore
import pytest

from mipengine.datatypes import DType
from mipengine.udfgen.decorator import UDFBadDefinition
from mipengine.udfgen.decorator import udf
from mipengine.udfgen.iotypes import merge_transfer
from mipengine.udfgen.iotypes import relation
from mipengine.udfgen.iotypes import secure_transfer
from mipengine.udfgen.iotypes import state
from mipengine.udfgen.iotypes import tensor
from mipengine.udfgen.iotypes import transfer
from mipengine.udfgen.iotypes import udf_logger


class TestUDFValidation:
    def test_validate_func_as_udf_invalid_input_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor, return_type=relation([("result", int)]))
            def f_invalid_sig(x):
                x = 1
                return x

        assert "Input types of func are not subclasses of InputType" in str(exc)

    def test_validate_func_as_udf_invalid_output_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=bool)
            def f_invalid_sig(x):
                x = 1
                return x

        assert "Output type of func is not subclass of OutputType" in str(exc)

    def test_validate_func_as_udf_invalid_expression_in_return_stmt(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=relation([("result", int)]))
            def f_invalid_ret_stmt(x):
                x = 1
                return x + 1

        assert "Expression in return statement" in str(exc)

    def test_validate_func_as_udf_invalid_no_return_stmt(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=relation([("result", int)]))
            def f_invalid_ret_stmt(x):
                pass

        assert "Return statement not found" in str(exc)

    def test_validate_func_as_udf_invalid_parameter_names(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(y=tensor(int, 1), return_type=relation([("result", int)]))
            def f(x):
                return x

        assert "The parameters: y were not provided in the func definition." in str(exc)

    def test_validate_func_as_udf_undeclared_parameter_names(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(y=tensor(int, 1), return_type=relation([("result", int)]))
            def f(y, x):
                return x

        assert "The parameters: x were not defined in the decorator." in str(exc)

    def test_validate_func_as_udf_no_return_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1))
            def f(x):
                return x

        assert "No return_type defined." in str(exc)

    def test_validate_func_as_valid_udf_with_state_and_transfer_input(self):
        @udf(
            x=tensor(int, 1),
            y=state(),
            z=transfer(),
            return_type=relation([("result", int)]),
        )
        def f(x, y, z):
            return x

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_transfer_output(self):
        @udf(x=tensor(int, 1), return_type=transfer())
        def f(x):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_state_output(self):
        @udf(
            x=state(),
            y=transfer(),
            return_type=transfer(),
        )
        def f(x, y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_merge_transfer_input(self):
        @udf(
            x=state(),
            y=merge_transfer(),
            return_type=state(),
        )
        def f(x, y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_local_step_logic_and_state_main_return(
        self,
    ):
        @udf(x=state(), y=transfer(), return_type=[state(), transfer()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_local_step_logic_and_transfer_main_return(
        self,
    ):
        @udf(x=state(), y=transfer(), return_type=[transfer(), state()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_global_step_logic(self):
        @udf(x=state(), y=merge_transfer(), return_type=[state(), transfer()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_invalid_udf_with_tensor_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(),
                y=merge_transfer(),
                return_type=[state(), tensor(DType.INT, 2)],
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

    def test_validate_func_as_invalid_udf_with_relation_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(),
                y=merge_transfer(),
                return_type=[state(), relation(schema=[])],
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

    def test_validate_func_as_invalid_udf_with_scalar_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(),
                y=merge_transfer(),
                return_type=[state(), relation([("result", int)])],
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

    def test_tensors_and_relations(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=tensor(dtype=int, ndims=1),
                y=relation(schema=[]),
                return_type=relation([("result", int)]),
            )
            def f(x, y):
                return x

        assert "tensors and relations" in str(exc)

    def test_validate_func_as_valid_udf_with_secure_transfer_output(self):
        @udf(
            y=state(),
            return_type=secure_transfer(sum_op=True),
        )
        def f(y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_secure_transfer_input(self):
        @udf(
            y=secure_transfer(sum_op=True),
            return_type=transfer(),
        )
        def f(y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_logger_input(self):
        @udf(
            y=transfer(),
            logger=udf_logger(),
            return_type=transfer(),
        )
        def f(y, logger):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_invalid_if_logger_is_not_the_last_input_parameter(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                logger=udf_logger(),
                y=transfer(),
                return_type=transfer(),
            )
            def f(logger, y):
                y = {"num": 1}
                return y

        assert "'udf_logger' must be the last input parameter" in str(exc)

    def test_validate_func_as_invalid_if_logger_exists_more_than_once(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                y=transfer(),
                logger1=udf_logger(),
                logger2=udf_logger(),
                return_type=transfer(),
            )
            def f(y, logger1, logger2):
                y = {"num": 1}
                return y

        assert "Only one 'udf_logger' parameter can exist" in str(exc)
