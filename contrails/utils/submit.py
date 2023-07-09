import os
from pathlib import Path

import numpy as np
import pandas as pd

from contrails.utils import data_utils

data_path = Path("/media/master/wext/projects/contrails/data")


def get_subm_df(test_recs, n=1000):
    submission = pd.read_csv(data_path / "sample_submission.csv", index_col="record_id")

    for rec in test_recs:
        band_08 = np.load(data_path / "test" / rec / "band_08.npy").sum(axis=2)
        preds = np.c_[
            np.unravel_index(np.argpartition(band_08.ravel(), -n)[-n:], band_08.shape)
        ]
        mask = np.zeros((256, 266))
        mask[preds[:, 0], preds[:, 1]] = 1
        # notice the we're converting rec to an `int` here:
        submission.loc[int(rec), "encoded_pixels"] = data_utils.list_to_string(
            data_utils.rle_encode(mask)
        )

    print(submission)


if __name__ == "__main__":
    test_recs = os.listdir(data_path / "test")
    get_subm_df(test_recs)
