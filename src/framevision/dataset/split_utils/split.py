from pathlib import Path
from typing import Dict, List, Set, Union

# A Split is a dictionary where:
# - The keys are "train", "val", and "test", each representing a different data split.
# - The value for each key is a dictionary where:
#     - The keys are sequence names.
#     - The value for each sequence name is a list of action names included in that sequence.
Split = Dict[str, Dict[str, List[str]]]


def split_by_actions(root_path: Path, split: Dict[str, Union[List[str], str]]) -> Split:
    all_sequences = get_all_sequences(root_path)
    all_actions = get_all_actions(root_path, all_sequences)

    train_actions = set(split.get("train", []))
    val_actions = set(split.get("val", []))
    test_actions = set(split.get("test", []))

    if split.get("train") == "others":
        train_actions = all_actions - val_actions - test_actions

    train_data, val_data, test_data = {}, {}, {}

    for seq in all_sequences:
        seq_actions = get_actions(root_path, seq)

        seq_train = sorted(list(seq_actions & train_actions))
        seq_val = sorted(list(seq_actions & val_actions))
        seq_test = sorted(list(seq_actions & test_actions))

        if seq_train:
            train_data[seq] = seq_train
        if seq_val:
            val_data[seq] = seq_val
        if seq_test:
            test_data[seq] = seq_test

    return {"train": train_data, "val": val_data, "test": test_data}


def split_by_sequences(root_path: Path, split: Dict[str, Union[List[str], str]]) -> Split:
    all_sequences = get_all_sequences(root_path)

    train_seqs = set(split.get("train", []))
    val_seqs = set(split.get("val", []))
    test_seqs = set(split.get("test", []))

    if split.get("train") == "others":
        train_seqs = set(all_sequences) - val_seqs - test_seqs

    return {
        "train": {seq: sorted(list(get_actions(root_path, seq))) for seq in sorted(list(train_seqs))},
        "val": {seq: sorted(list(get_actions(root_path, seq))) for seq in sorted(list(val_seqs))},
        "test": {seq: sorted(list(get_actions(root_path, seq))) for seq in sorted(list(test_seqs))},
    }


def split_by_both(root_path: Path, split: Dict[str, Union[List[str], str]]) -> Split:
    all_sequences = get_all_sequences(root_path)

    train_data = split.get("train", {})
    val_data = split.get("val", {})
    test_data = split.get("test", {})

    # Fill in the missing sequences if "others" is specified
    if split.get("train") == "others":
        train_data = {seq: get_actions(root_path, seq) for seq in all_sequences}

        for seq, actions in (val_data | test_data).items():
            if seq not in train_data:
                continue

            train_data["seq"] = train_data["seq"] - set(actions)

        train_data = {seq: sorted(list(actions)) for seq, actions in train_data.items()}

    # Fill in the missing actions if "others" is specified
    for seq in train_data:
        if len(train_data[seq]) == 1 and train_data[seq][0] == "others":
            existing_actions = set(val_data.get(seq, [])) - set(test_data.get(seq, []))
            train_data[seq] = sorted(list(get_actions(root_path, seq)) - existing_actions)

    return {"train": train_data, "val": val_data, "test": test_data}


def get_actions(root_path: Path, sequence: str) -> Set[str]:
    return set([f.name for f in (root_path / sequence / "actions").iterdir() if f.is_dir()])


def get_all_actions(root_path: Path, sequences: List[str]) -> Set[str]:
    all_actions = set()
    for seq in sequences:
        seq_actions = set(get_actions(root_path, seq))
        all_actions.update(seq_actions)
    return all_actions


def get_all_sequences(root_path: Path) -> List[str]:
    sequences = [f.name for f in root_path.iterdir() if f.is_dir() and f.name != ".cache"]
    return sorted(sequences)
