# Python Version Alignment Guide

## Issue
- **Local environment**: Python 3.14.2
- **Dockerfile**: Python 3.11-slim
- **Risk**: Compatibility issues between dev and production

## Recommendation
Downgrade local environment to **Python 3.11** to match production.

## Steps to Fix

### Option 1: Using pyenv (Recommended)

```bash
# Install Python 3.11 (latest 3.11.x)
pyenv install 3.11.9

# Set local version for this project
cd C:\oper\me\langchain-agent\agent
pyenv local 3.11.9

# Verify
python --version  # Should show Python 3.11.9

# Recreate virtual environment
rm -rf .venv
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

### Option 2: Using Python Installer

1. Download Python 3.11.9 from https://www.python.org/downloads/release/python-3119/
2. Install (don't uninstall 3.14, you can have multiple versions)
3. Recreate venv with Python 3.11:

```bash
cd C:\oper\me\langchain-agent\agent

# Remove old venv
Remove-Item -Recurse -Force .venv

# Create new venv with Python 3.11
py -3.11 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Option 3: Using conda

```bash
# Create new environment with Python 3.11
conda create -n langchain-agent-311 python=3.11

# Activate
conda activate langchain-agent-311

# Install dependencies
cd C:\oper\me\langchain-agent\agent
pip install -r requirements.txt
```

## Verification

After switching to Python 3.11:

```bash
# Check Python version
python --version
# Expected: Python 3.11.x

# Run tests
pytest tests/ -v

# Run server
python server.py

# Run ingestion
python cronjob.py
```

## Why Not Upgrade Dockerfile to 3.14?

1. **Stability**: Python 3.14 is bleeding edge (released 2026)
2. **Package compatibility**: Some packages may not be fully tested on 3.14
3. **Production risk**: Changing production Python version requires extensive testing
4. **No benefit**: The codebase doesn't use Python 3.14-specific features

## Alternative: Upgrade to Python 3.12 or 3.13

If you want a newer version while maintaining stability:

```bash
# Update Dockerfile
FROM python:3.12-slim  # or 3.13-slim

# Update local environment
pyenv install 3.12.7  # or 3.13.0
pyenv local 3.12.7
```

**Recommendation**: Stick with 3.11 for now. Upgrade to 3.12/3.13 in a future release after testing.

## Future Maintenance

When upgrading Python versions:
1. Test thoroughly in development first
2. Run full test suite: `pytest tests/ -v`
3. Update Dockerfile and local environment together
4. Monitor for deprecation warnings
5. Update CI/CD pipelines if applicable
