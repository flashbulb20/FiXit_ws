import os

def remap_yolo_labels(folder_path):

    class_map = {
        15: 1,
        16: 4,
        17: 3,
        18: 2
    }

    for filename in os.listdir(folder_path):
        if not filename.startswith("tools_") or not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)

        new_lines = []
        modified = False

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                new_lines.append(line)
                continue

            cls_id = int(parts[0])

            if cls_id in class_map:
                parts[0] = str(class_map[cls_id])
                modified = True

            new_lines.append(" ".join(parts) + "\n")

        if modified:
            with open(file_path, "w") as f:
                f.writelines(new_lines)

            print(f"[UPDATED] {filename}")
        else:
            print(f"[SKIPPED] {filename} (no target labels)")

remap_yolo_labels("/home/hyunj/labels")