import argparse
from xdsl.interactive.app import InputApp

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms import get_all_passes


def main() -> int:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "input_file", type=str, nargs="?", help="path to input file"
    )

    # available_passes = ",".join([name for name in get_all_passes()])
    # arg_parser.add_argument(
    #     "-p",
    #     "--passes",
    #     required=False,
    #     help="Delimited list of passes." f" Available passes are: {available_passes}",
    #     type=str,
    #     default="",
    # )
    args = arg_parser.parse_args()

    file_path = args.input_file
    if file_path is not None:
        # Open the file and read its contents
        with open(file_path) as file:
            file_contents = file.read()
    else:
        file_contents = None

    # pass_spec_pipeline = list(parse_pipeline(args.passes))
    # pass_list = get_all_passes()
    # pipeline = tuple(PipelinePass.build_pipeline_tuples(pass_list, pass_spec_pipeline))

    app = InputApp(
        tuple(get_all_dialects().items()),
        tuple((p_name, p()) for p_name, p in sorted(get_all_passes().items())),
        file_path,
        file_contents,
    )
    result: int = 0
    run_method = getattr(app, 'run', None)
    if callable(run_method):
        maybe_result = run_method()
        if isinstance(maybe_result, int):
            result = maybe_result
    return result


if __name__ == "__main__":
    main()
