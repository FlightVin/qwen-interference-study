"""Tag superseded W&B runs (A11): add tag `superseded-A11` and a note to the given run ids; nothing is deleted.
Usage: python -m qis.analysis.wandb_tag_superseded ENTITY/PROJECT RUN_ID [RUN_ID ...]"""
import sys


def main(path, *ids):
    import wandb
    api = wandb.Api()
    for rid in ids:
        try:
            r = api.run(f"{path}/{rid}")
        except Exception as e:  # noqa: BLE001
            print("missing", rid, str(e)[:120]); continue
        if "superseded-A11" not in r.tags:
            r.tags = r.tags + ["superseded-A11"]
        r.notes = (r.notes or "") + " Superseded by amendment A11 (cap transition moved to round 32; trajectory replayed)."
        r.update()
        print("tagged", rid)


if __name__ == "__main__":
    main(*sys.argv[1:])
