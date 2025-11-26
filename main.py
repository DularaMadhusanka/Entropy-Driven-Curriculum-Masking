from arguments import get_args
from runs import RUNS
import sys


def _print_available_and_exit(model_name, dataset):
    available_models = sorted(RUNS.keys())
    msg = [f"Unknown model '{model_name}' or dataset '{dataset}' provided."]
    msg.append('Available models:')
    for m in available_models:
        datasets = sorted(RUNS[m].keys())
        msg.append(f"  - {m}: {', '.join(datasets)}")
    print('\n'.join(msg), file=sys.stderr)
    sys.exit(2)


if __name__ == '__main__':
    args = get_args()
    model = getattr(args, 'model_name', None)
    dataset = getattr(args, 'dataset', None)
    if model not in RUNS or dataset not in RUNS.get(model, {}):
        _print_available_and_exit(model, dataset)
    RUNS[model][dataset](args)