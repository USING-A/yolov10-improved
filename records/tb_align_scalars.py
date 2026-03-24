import argparse
import os
import shutil
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def load_scalars(logdir):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    scalars = {t: ea.Scalars(t) for t in tags}
    return scalars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base run logdir, e.g. e100 run")
    parser.add_argument("--cont", required=True, help="Continuation run logdir, e.g. +30 run")
    parser.add_argument("--out", required=True, help="Output aligned logdir")
    args = parser.parse_args()

    base = load_scalars(args.base)
    cont = load_scalars(args.cont)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    max_base_step = 0
    for events in base.values():
        if events:
            max_base_step = max(max_base_step, int(max(e.step for e in events)))

    writer = SummaryWriter(args.out)

    all_tags = sorted(set(base.keys()) | set(cont.keys()))
    for tag in all_tags:
        for e in base.get(tag, []):
            writer.add_scalar(tag, e.value, int(e.step))
        for e in cont.get(tag, []):
            writer.add_scalar(tag, e.value, int(e.step) + max_base_step)

    writer.add_text("alignment/info", f"base={args.base}; cont={args.cont}; step_offset={max_base_step}", 0)
    writer.close()

    print(f"aligned_logdir={args.out}")
    print(f"step_offset={max_base_step}")


if __name__ == "__main__":
    main()
