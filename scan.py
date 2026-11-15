import os

print("\n=== PEMINDAIKAN PERKALIAN STRING DI PROYEK ===")
found = 0
for root, dirs, files in os.walk('.'):
    if 'venv' in root or '.git' in root:
        continue
    for file in files:
        if file.endswith('.py') and file != 'scan.py':
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    # Deteksi baris yang menggunakan perkalian string
                    if '*' in line and ('"' in line or "'" in line) and not stripped.startswith('#'):
                        print(f"{filepath} [Baris {line_num}]: {stripped}")
                        found += 1

print(f"\nSelesai! Ditemukan {found} baris potensi perkalian string.\n")