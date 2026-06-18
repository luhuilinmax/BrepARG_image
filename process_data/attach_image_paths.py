import argparse
import copy
import os
import pickle


def load_render_index(index_path):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "renders" in data:
        return data["renders"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported render index format: {index_path}")


def stem_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def attach_to_split(split_data, render_index, strict=False):
    output = {}
    missing = []
    for split_name in ("train", "val", "test"):
        items = split_data.get(split_name, [])
        enriched = []
        for item in items:
            stem = stem_from_path(item)
            render_info = render_index.get(stem)
            if render_info is None:
                missing.append({"split": split_name, "stem": stem, "source": item})
                if strict:
                    raise KeyError(f"Missing render for {stem}")
                continue
            enriched.append(
                {
                    "cad_path": item,
                    "cad_stem": stem,
                    "image_path": render_info["image_path"] if isinstance(render_info, dict) else render_info,
                    "render_info": render_info if isinstance(render_info, dict) else {"image_path": render_info},
                }
            )
        output[split_name] = enriched
    return output, missing


def attach_to_sequences(sequence_data, render_index, split_entries=None, strict=False):
    output = copy.deepcopy(sequence_data)
    missing = []
    if split_entries is None:
        raise ValueError("Sequence attachment requires --split_input to map each group to a CAD stem.")

    for split_name in ("train", "val", "test"):
        groups = output.get(split_name, [])
        entries = split_entries.get(split_name, [])
        if len(groups) != len(entries):
            raise ValueError(
                f"Split size mismatch for {split_name}: sequence has {len(groups)} groups, split has {len(entries)} entries"
            )
        for idx, (group, entry) in enumerate(zip(groups, entries)):
            source_path = entry["cad_path"] if isinstance(entry, dict) else entry
            stem = entry["cad_stem"] if isinstance(entry, dict) else stem_from_path(source_path)
            render_info = render_index.get(stem)
            if render_info is None:
                missing.append({"split": split_name, "index": idx, "stem": stem, "source": source_path})
                if strict:
                    raise KeyError(f"Missing render for {stem}")
                continue
            group["cad_stem"] = stem
            group["cad_path"] = source_path
            group["image_path"] = render_info["image_path"] if isinstance(render_info, dict) else render_info
            group["render_info"] = render_info if isinstance(render_info, dict) else {"image_path": render_info}
    return output, missing


def main():
    parser = argparse.ArgumentParser(description="Attach rendered image paths to split or sequence pickle files.")
    parser.add_argument("--render_index", type=str, required=True, help="Pickle index created by render_step_images.py")
    parser.add_argument("--mode", type=str, choices=["split", "sequence"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Input split/sequence pickle")
    parser.add_argument("--output", type=str, required=True, help="Output pickle with image paths attached")
    parser.add_argument("--split_input", type=str, default="", help="Split pickle or enriched split pickle used to align sequence groups")
    parser.add_argument("--strict", action="store_true", help="Fail if any render is missing")
    args = parser.parse_args()

    render_index = load_render_index(args.render_index)
    with open(args.input, "rb") as f:
        input_data = pickle.load(f)

    if args.mode == "split":
        output_data, missing = attach_to_split(input_data, render_index, strict=args.strict)
    else:
        split_entries = None
        if args.split_input:
            with open(args.split_input, "rb") as f:
                split_entries = pickle.load(f)
        output_data, missing = attach_to_sequences(
            input_data,
            render_index,
            split_entries=split_entries,
            strict=args.strict,
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print(f"Saved: {args.output}")
    print(f"Missing renders: {len(missing)}")
    if missing:
        print("First missing:", missing[0])


if __name__ == "__main__":
    main()
