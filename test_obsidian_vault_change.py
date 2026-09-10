from pathlib import Path
p = Path("/Users/dariomarcolin/Library/Mobile Documents/iCloud~md~obsidian/Documents")
md_files = list(p.rglob("*.md"))
print(len(md_files))
print(md_files[:5])