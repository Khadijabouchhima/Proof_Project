import json
from pathlib import Path

PMDATA = Path("pmdata")

FILES = [
    "fitbit/resting_heart_rate.json",
    "fitbit/heart_rate.json",
    "fitbit/steps.json",
    "fitbit/sleep.json",
]

participants = sorted(
    p for p in PMDATA.iterdir()
    if p.is_dir() and p.name.startswith("p")
)

print("=" * 60)
print("CERTAIN PMData AUDIT")
print("=" * 60)

print(f"Participants: {len(participants)}")
print("Names:", ", ".join(p.name for p in participants))

print("\nFILE COVERAGE")
print("-" * 60)

for file in FILES:
    present = []
    missing = []

    for p in participants:
        path = p / file

        if path.exists():
            present.append(p.name)
        else:
            missing.append(p.name)

    print(f"\n{file}")
    print(f"  Present: {len(present)}/{len(participants)}")

    if missing:
        print("  Missing:", ", ".join(missing))
    else:
        print("  Missing: NONE")


print("\nDATASET SIZES")
print("-" * 60)

for file in FILES:

    total_records = 0
    errors = []

    for p in participants:
        path = p / file

        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                total_records += len(data)

        except Exception as e:
            errors.append(f"{p.name}: {e}")

    print(f"{file}: {total_records:,} total records")

    if errors:
        print("  ERRORS:", len(errors))
        for error in errors:
            print("   ", error)


print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
