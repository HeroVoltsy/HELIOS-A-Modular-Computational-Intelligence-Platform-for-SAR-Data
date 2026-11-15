
from pathlib import Path
import h5py

DATA_DIR = Path("data/nasa_nisar")

files = list(DATA_DIR.glob("*.h5"))

if not files:
    raise FileNotFoundError("No NISAR HDF5 granule found.")

h5_file = files[0]

print("=" * 90)
print("NISAR GCOV HDF5 INSPECTION")
print("=" * 90)
print("File:", h5_file.name)
print(f"Size: {h5_file.stat().st_size / 1e9:.3f} GB")

with h5py.File(h5_file, "r") as f:

    print("\nROOT ATTRIBUTES")
    for key, value in f.attrs.items():
        print(f"{key}: {value}")

    print("\nDATASET HIERARCHY")

    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(
                f"\nPATH: /{name}"
                f"\nSHAPE: {obj.shape}"
                f"\nDTYPE: {obj.dtype}"
                f"\nCHUNKS: {obj.chunks}"
            )

            for key, value in obj.attrs.items():
                print(f"  ATTR {key}: {value}")

    f.visititems(inspect)

    print("\nGROUP ATTRIBUTES")

    def inspect_groups(name, obj):
        if isinstance(obj, h5py.Group) and len(obj.attrs):
            print(f"\nGROUP: /{name}")
            for key, value in obj.attrs.items():
                print(f"  {key}: {value}")

    f.visititems(inspect_groups)

print("\nINSPECTION COMPLETE")
