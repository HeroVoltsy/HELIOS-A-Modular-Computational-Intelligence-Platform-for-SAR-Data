
from pathlib import Path
import earthaccess

OUTPUT = Path("data/nasa_nisar")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Approximate Philippines bounding box:
# west, south, east, north
BBOX = (116.9, 4.5, 126.6, 21.3)

auth = earthaccess.login()

search_configs = [
    {
        "name": "VALIDATED",
        "short_name": "NISAR_L2_GCOV_V1",
        "temporal": ("2026-10-21", "2026-11-16"),
    },
    {
        "name": "PROVISIONAL",
        "short_name": "NISAR_L2_GCOV_PROVISIONAL_V1",
        "temporal": ("2026-06-17", "2026-10-23"),
    },
]

results = []
selected_maturity = None

for config in search_configs:
    print(f"\nSearching {config['name']} GCOV products...")

    results = earthaccess.search_data(
        short_name=config["short_name"],
        bounding_box=BBOX,
        temporal=config["temporal"],
        count=10,
    )

    if results:
        selected_maturity = config["name"]
        break

if not results:
    raise RuntimeError(
        "No GCOV granules found in the Philippines "
        "for the selected time windows. "
        "Try expanding the geographic or temporal search."
    )

print(f"\nFound {len(results)} granules.")
print(f"Selected maturity: {selected_maturity}")

for i, result in enumerate(results):
    print(f"\n[{i}]")
    print(result)

# Download only one granule for the initial prototype.
print("\nDownloading first granule...")

downloaded = earthaccess.download(
    [results[0]],
    local_path=str(OUTPUT),
)

print("\nDOWNLOAD COMPLETE")
print("Files:", downloaded)

with open(OUTPUT / "source_manifest.txt", "w") as f:
    f.write("Mission: NASA-ISRO NISAR\n")
    f.write("Product: L2 GCOV\n")
    f.write(f"Maturity: {selected_maturity}\n")
    f.write(f"Granule: {results[0]}\n")
    f.write(f"Files: {downloaded}\n")
