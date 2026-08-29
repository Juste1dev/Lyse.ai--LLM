from __future__ import annotations

import argparse
import json
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(description="Convertit un export JSONL vers le format messages attendu")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt-key", type=str, default="prompt")
    parser.add_argument("--response-key", type=str, default="response")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if "messages" in row:
                obj = {"messages": row["messages"]}
            else:
                obj = {
                    "messages": [
                        {"role": "user", "content": row.get(args.prompt_key, "")},
                        {"role": "assistant", "content": row.get(args.response_key, "")},
                    ]
                }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
