"""Dataset preprocessing — convert various JSONL formats to Axolotl chat_template format."""

import json


def preprocess_dataset(input_path, output_path):
    """
    Convert JSONL to Axolotl chat_template format.

    Supports input formats:
      A) {"system": "...", "user": "...", "assistant": "..."}
      B) {"messages": [{"role": "system", "content": "..."}, ...]}
      C) {"conversations": [{"role": "system", "content": "..."}, ...]}
         (also handles sharegpt's {"from"/"value"} variant)

    Output format (one per line):
      {"messages": [{"role":"system","content":"..."},
                     {"role":"user","content":"..."},
                     {"role":"assistant","content":"..."}]}

    Returns:
        Number of examples successfully processed.
    """
    print(f"Preprocessing dataset: {input_path}")
    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {line_num} (invalid JSON): {e}")
                continue

            # Format A: flat keys
            if "system" in row and "user" in row and "assistant" in row:
                messages = [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": row["assistant"]},
                ]
            # Format B: messages list
            elif "messages" in row:
                messages = row["messages"]
            # Format C: conversations list
            elif "conversations" in row:
                messages = row["conversations"]
                # Normalize from/value -> role/content if needed
                if messages and "from" in messages[0]:
                    role_map = {
                        "system": "system", "human": "user", "gpt": "assistant",
                        "user": "user", "assistant": "assistant",
                    }
                    messages = [
                        {"role": role_map.get(m["from"], m["from"]),
                         "content": m.get("value", m.get("content", ""))}
                        for m in messages
                    ]
            else:
                print(f"  Warning: skipping line {line_num} (unrecognized format)")
                continue

            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    print(f"  Processed {count} examples -> {output_path}")
    return count
