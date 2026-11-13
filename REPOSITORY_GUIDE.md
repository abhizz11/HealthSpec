# Source-only publication

The publication selection contains seven Python source files plus `.gitignore`,
`.gitattributes`, `README.md`, `requirements.txt`, and this guide: 12 files total.
The README describes the seven source files and their roles.

All experiment folders, raw records, policies, results, figures, paper files,
caches, logs, exploratory comparisons and scratch scripts stay local and ignored.
Nothing was deleted or moved. `.gitattributes` preserves source bytes by preventing
automatic newline conversion.

Create an empty repository on GitHub, without an automatically generated README,
license or gitignore. Replace the username and repository name below:

```powershell
Set-Location 'C:\Users\fisk_\OneDrive\Desktop\Speculative Decoding'
git init -b main
git add --dry-run .
git add .
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add HealthSpec Python implementation"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Review the staged paths before committing. These commands assume a new local Git
repository. If records were previously staged or tracked, ignore rules alone do
not remove them from Git's index.

No Git repository, commit, remote or push was created while preparing these files.
