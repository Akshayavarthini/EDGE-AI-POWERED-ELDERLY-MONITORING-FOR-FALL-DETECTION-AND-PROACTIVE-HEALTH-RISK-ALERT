# convert_to_header.py
import sys

def convert_tflite_to_header(tflite_file, header_file, var_name="health_model"):
    with open(tflite_file, "rb") as f:
        tflite_model = f.read()

    with open(header_file, "w") as f:
        f.write(f'const unsigned char {var_name}[] = {{\n')

        for i, val in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("\n  ")
            f.write(f"0x{val:02x}, ")

        f.write("\n};\n")
        f.write(f'const int {var_name}_len = {len(tflite_model)};\n')

    print(f"✅ Converted {tflite_file} → {header_file}")

if __name__ == "__main__":
    convert_tflite_to_header("health_model.tflite", "helmet_model.h")
