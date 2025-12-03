import numpy as np
import zarr

store = "foo.zarr"

np_dtype = np.dtype(
    [
        ("field_1", "<i4"),
        ("field_2", ">i4"),
        ("field_n", "<f4"),
    ]
)

z = zarr.create_array(
    store=store,
    shape=(128, 128),
    dtype=np_dtype,
    chunks=(32, 32),
)

arr = np.zeros((128, 128), dtype=np_dtype)

f1 = np.arange(128, dtype="<i4")
# NOTE: We populate f2 as little endian for the demonstration
f2 = np.arange(128, dtype="<i4")
fn = np.arange(128, dtype="<f4") / 10

arr["field_1"][:] = f1[:, None]
arr["field_2"][:] = f2[:, None]
arr["field_n"][:] = fn[:, None]

z[:] = arr
