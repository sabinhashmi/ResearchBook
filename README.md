# Project: **ResearchBook**

## 🔢 Versions

- **Moore**: `v58r2`
- **Rec**: `v39r2`
- **DaVinci**: _not specified_
- **ParamFiles**: _not specified_

---

## 🖥 Platform

- `x86_64_v3-el9-gcc13+detdesc-opt+g`

---

## 📁 Repository Structure

- The inner Git repository’s `.git` directory has been renamed to `.git_backup` to **prevent it from being treated as a Git submodule** when nested inside this main repository.
- This preserves the inner repository's Git metadata for **independent version control**, while keeping it decoupled from the parent project.
- The outer repository **only tracks its own files** and does not manage the inner repo as a submodule.

---

## 📄 `.gitignore` Notes

- The renamed `.git_backup` directories should be explicitly **ignored** in the `.gitignore` to prevent accidental commits.
- Ensure consistent naming of such internal Git folders across collaborators for clarity and automation.

---

## ⚙️ Pre-requisites

Run the following setup commands in order:

```bash
# Probe CVMFS if not already done
cvmfs_config probe

# Load the LHCb environment
source /cvmfs/lhcb.cern.ch/lib/LbEnv

# Initialize proxy to access DIRAC
lhcb-proxy-init

# Example Run Command

../../Moore/Baseline/run gaudirun.py ../../Options/TemplateOptions.py | tee baseline.log


## Working Directory consists of new approach of downstream track classifier
