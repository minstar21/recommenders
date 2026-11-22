"""Synthetic SAR oracle using standard-library arithmetic, independent of SARPlus.

Cooccurrence counts common users; Jaccard divides by union size; lift divides
by frequencies. Affinity halves every 30 days; scores are affinity x similarity.
"""
import csv
from datetime import datetime, timedelta
from pathlib import Path


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    target = "0003000098E85347"
    users = [target] + [f"user{i}" for i in range(1, 9)]
    items = [f"item{i:02}" for i in range(1, 15)]
    events = []
    start = datetime(2024, 1, 1)
    for u, user in enumerate(users):
        for i, item in enumerate(items):
            dense = i < 12 and ((u == 0 and i < 3) or (u > 0 and (u * 7 + i * 3) % 11 < 8))
            rare = (i == 12 and u == 8) or (i == 13 and u in (7, 8))
            if dense or rare:
                events.append((user, item, start + timedelta(days=u * 2 + i)))
    now = max(t for _, _, t in events)
    memberships = {item: {u for u, i, _ in events if i == item} for item in items}
    affinity = {
        item: sum(2 ** ((t - now).total_seconds() / (30 * 86400))
                  for u, i, t in events if u == target and i == item)
        for item in items
    }

    def write(name, header, rows):
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(header)
            out.writerows(rows)

    write("demoUsage.csv", ["userId", "productId", "timestamp"],
          [(u, i, t.strftime("%Y/%m/%dT%H:%M:%S")) for u, i, t in events])
    for threshold in (1, 3):
        for kind in ("count", "jac", "lift"):
            matrix = {}
            for a in items:
                for b in items:
                    common = len(memberships[a] & memberships[b])
                    value = common if common >= threshold else 0
                    if kind == "jac":
                        value /= len(memberships[a] | memberships[b])
                    if kind == "lift":
                        value /= len(memberships[a]) * len(memberships[b])
                    matrix[a, b] = value
            write(f"sim_{kind}{threshold}.csv", ["item"] + items,
                  [[a] + [matrix[a, b] for b in items] for a in items])
            if threshold == 3:
                scores = [(b, sum(affinity[a] * matrix[a, b] for a in items))
                          for b in items if affinity[b] == 0]
                scores = sorted((p for p in scores if p[1] > 0), key=lambda p: -p[1])[:10]
                header = ["user"] + [
                    v for n in range(1, len(scores) + 1)
                    for v in (f"rec{n}", f"score{n}")
                ]
                write(f"userpred_{kind}3_userid_only.csv", header,
                      [[target] + [v for pair in scores for v in pair]])
    write("user_aff.csv", ["user"] + items, [[target] + [affinity[i] for i in items]])
    write("user_aff_2_months_later.csv", ["item", "affinity"],
          [[i, affinity[i] / 4] for i in items])


if __name__ == "__main__":
    generate(Path(__file__).parent / "data")
