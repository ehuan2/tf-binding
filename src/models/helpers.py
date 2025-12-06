import torch
import pandas as pd


def batch_to_pandas(data):
    """
    Convert a batch of data to a pandas DataFrame for MLFlow evaluation.
    """
    # first get the dataloader
    data = torch.utils.data.DataLoader(data, shuffle=False)

    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    all_df = []
    for batch in data:
        # first let's flatten the batch entry
        flat_batch = flatten_dict(batch)
        pd_dict = {}

        # then we need to convert these to numpy
        for key, val in flat_batch.items():
            if torch.is_tensor(val):
                flat_batch[key] = val.cpu().numpy()

                if val.ndim == 1:  # shape (1,)
                    pd_dict[key] = flat_batch[key]

                elif val.ndim == 2:  # shape (1, features)
                    for i in range(val.shape[1]):
                        pd_dict[f"{key}.{i}"] = flat_batch[key][:, i]
            else:
                # this is the sequencing part
                pd_dict[key] = val

        all_df.append(pd.DataFrame(pd_dict))

    return pd.concat(all_df, ignore_index=True)


def pandas_to_batch(df: pd.DataFrame, config):
    # implement the logic to convert pandas dataframe to the model input format
    # the opposite of batch_to_pandas
    # First, split multi-feature keys (ending with .idx or _idx)
    temp = {}
    sep = "."
    for col in df.columns:
        if sep in col and col.rsplit(sep, 1)[-1].isdigit():
            key_base, idx = col.rsplit(sep, 1)

            if temp.get(key_base) is None:
                temp[key_base] = []
            temp[key_base].append((int(idx), col))
        else:
            if temp.get(col) is None:
                temp[col] = []
            temp[col].append((None, col))

    # Now build the nested dict
    def set_nested(d, keys, value):
        """Recursively set value in nested dict given list of keys."""
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    batch = {}
    for key_base, cols_info in temp.items():
        # Sort by idx if present
        cols_info.sort(key=lambda x: (x[0] if x[0] is not None else -1))
        keys = key_base.split(".")
        if cols_info[0][0] is not None:
            # Multi-feature tensor
            arr = df[[col for _, col in cols_info]].to_numpy()
            tensor = torch.tensor(arr, device=config.device, dtype=config.dtype)
        else:
            # Single feature
            if df[cols_info[0][1]].dtype == object:
                # likely string data
                tensor = df[cols_info[0][1]].tolist()
            else:
                tensor = torch.tensor(
                    df[cols_info[0][1]].to_numpy(),
                    device=config.device,
                    dtype=config.dtype,
                )
        set_nested(batch, keys, tensor)

    return batch
