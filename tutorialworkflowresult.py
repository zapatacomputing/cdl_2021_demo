import json
import os
import typing
import codecs
import typing
import os
import json
import dill
from dataclasses import dataclass, field

ENCODED_PICKLE = "encodedpickle"


class TutorialJsonIOManager(typing.List[str]):
    """
    TutorialJsonIOManager will read step results from a json file
    """

    def __init__(
        self,
        # NOTE Encoding scheme is a union of literals, not of type str
        # but typing.Literal does not exist in python 3.7
        # allow any type for now, to avoid mypy errors
        encoding_scheme="base64",
        file_name: str = "result.json",
        field_name: str = "result",
    ):
        self.encoding_scheme = encoding_scheme
        self.file_name = file_name
        self.field_name = field_name


    def deserialize(self, val: typing.Any) -> typing.Any:
        """
        If val is type List[str], then assume it is an encoded pickle
            (3) encode string as bytes (2) base64 decode (1) unpickle to python object
        Else if val is type str and of format `/app/_.json`
            first read the data from the file and run deserialize on contents
        Else assume it is just a raw value
            return val
        """

        def is_type_list_str(val: typing.Any) -> bool:
            if not isinstance(val, list):
                return False
            for element in val:
                if not isinstance(element, str):
                    return False
            return True

        # if val is type List[str], then assume it is an encoded pickle
        if is_type_list_str(val):
            val = "".join(val)
            return dill.loads(codecs.decode(val.encode(), self.encoding_scheme))

        # if val is `/app/_.json`, then it is a file we need to read before deserializing
        if (
            isinstance(val, str)
            and val.startswith(os.sep + "app")
            and val.endswith(".json")
        ):
            return self.read(val)

        # otherwise simply return val
        return val

    def read(self, file_path: str) -> typing.Any:
        """
        files must be valid json, so actual value is embedded in the result field
        e.g. {"type": "encodedpickle", "result": "__what we want as base64 encoded pickle__"}
        """

        with open(file_path, "r") as f:
            r = json.load(f)
            return self.deserialize(r[self.field_name])


@dataclass
class TaskResult:
    # map of input name to value
    inputs: typing.Dict[str, typing.Any] = field(default_factory=dict)
    result: typing.Optional[typing.Any] = None


@dataclass
class TutorialWorkflowResult:
    # map of task name to TaskResult
    tasks: typing.Dict[str, TaskResult] = field(default_factory=dict)

    def __str__(self) -> str:
        s = "\n" + " ┌" + "-" * (os.get_terminal_size().columns - 2) + "\n"
        for step_name, step_result in self.tasks.items():
            s += f" ├ {step_name:15s} : {step_result.result}\n"
        s += " └" + "-" * (os.get_terminal_size().columns - 2) + "\n"
        return s


def _deserialize_result(io_manager: TutorialJsonIOManager, result: typing.Dict[str, typing.Any]):
    if result["type"] == ENCODED_PICKLE:
        return io_manager.deserialize(result["result"])
    else:
        return result["result"]


def _deserialize_inputs(
    io_manager: TutorialJsonIOManager, inputs: typing.Dict, workflow_result_json: typing.Dict
):
    inputs_result = {}
    for k in inputs:
        # when input is raw value
        if "type" in inputs[k]:
            if inputs[k]["type"] == ENCODED_PICKLE:
                inputs_result[k] = io_manager.deserialize(inputs[k]["value"])
            else:
                inputs_result[k] = inputs[k]["value"]
        # when input is a result from a previous step
        elif "sourceArtifactName" in inputs[k]:
            inputs_result[k] = _deserialize_result(
                io_manager,
                workflow_result_json[inputs[k]["sourceStepID"]]["result"],
            )

    return inputs_result


def _load_workflowresult_from_dict(workflow_result_json: dict) -> TutorialWorkflowResult:
    io_manager = TutorialJsonIOManager()
    workflow_result = TutorialWorkflowResult()
    for step_id in workflow_result_json.keys():
        step = workflow_result_json[step_id]
        step_inputs = {}
        if "inputs" in step:
            step_inputs = _deserialize_inputs(
                io_manager, step["inputs"], workflow_result_json
            )
        step_result = None
        if "result" in step:
            step_result = _deserialize_result(io_manager, step["result"])

        workflow_result.tasks[step["stepName"]] = TaskResult(
            inputs=step_inputs,
            result=step_result,
        )
    return workflow_result


def load_cached_workflowresult(file_path: str) -> TutorialWorkflowResult:
    with open(file_path) as f:
        workflow_result_json = json.load(f)
        return _load_workflowresult_from_dict(workflow_result_json)
