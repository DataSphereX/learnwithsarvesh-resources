from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_dataset(size: int = 500) -> pd.DataFrame:
    rows = []
    channels = ["web", "mobile", "email"]
    regions = ["US", "IN", "EU", "APAC"]
    issues = ["account", "billing", "refund", "delivery", "subscription"]

    for idx in range(1, size + 1):
        issue = issues[idx % len(issues)]
        rows.append(
            {
                "question": f"How do I resolve {issue} issue #{idx}?",
                "answer": f"To resolve {issue}, open support ticket #{idx} from account settings and follow verification steps.",
                "region": regions[idx % len(regions)],
                "channel": channels[idx % len(channels)],
                "issue": issue,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "customer_support_faq.csv"
    df = build_dataset(size=500)
    df.to_csv(output_path, index=False)
    print(f"Generated FAQ CSV: {output_path}")


if __name__ == "__main__":
    main()
