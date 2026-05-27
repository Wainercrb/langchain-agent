"""
Complete project setup script.

This script handles:
1. Environment validation
2. Dependencies verification
3. Database schema setup
4. Directory creation
5. Configuration validation
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from services.container import logger


def check_python_version():
    """Verify Python version is 3.10+"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f"Python 3.10+ required, got {version.major}.{version.minor}")
        return False
    logger.info(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_env_file():
    """Verify .env file exists and contains required variables."""
    logger.info("Checking .env configuration...")

    env_file = Path(__file__).parent.parent / ".env"
    env_example = Path(__file__).parent.parent / ".env.example"

    if not env_file.exists():
        logger.warning(f"  .env not found at {env_file}")
        logger.info(f"  Creating from {env_example}...")

        if env_example.exists():
            with open(env_example, "r") as f:
                example_content = f.read()
            with open(env_file, "w") as f:
                f.write(example_content)
            logger.info(f"  ✓ Created {env_file}")
        else:
            logger.error(f"  {env_example} not found")
            return False

        logger.warning("  ⚠️  IMPORTANT: Edit .env and set these required variables:")
        logger.warning("     - GOOGLE_API_KEY (from Google AI Studio)")
        logger.warning("     - SUPABASE_URL (from Supabase project settings)")
        logger.warning("     - SUPABASE_KEY (from Supabase project settings)")
        return False

    # Read and validate .env
    required_vars = ["GOOGLE_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    env_vars = {}

    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")

    missing = [var for var in required_vars if not env_vars.get(var)]

    if missing:
        logger.error(f"  Missing required variables: {', '.join(missing)}")
        logger.info("  Edit .env and fill in the missing values")
        return False

    logger.info("  ✓ .env configured with all required variables")
    return True


def check_dependencies():
    """Verify all required packages are installed."""
    logger.info("Checking dependencies...")

    try:
        requirements_file = Path(__file__).parent.parent / "requirements-batch.txt"

        if not requirements_file.exists():
            logger.error(f"  {requirements_file} not found")
            return False

        # Try importing key dependencies
        dependencies = {
            "langchain": "LangChain",
            "fastapi": "FastAPI",
            "supabase": "Supabase",
            "apscheduler": "APScheduler",
            "pydantic": "Pydantic",
        }

        missing = []
        for module, name in dependencies.items():
            try:
                __import__(module)
                logger.info(f"  ✓ {name}")
            except ImportError:
                missing.append(f"{module} ({name})")

        if missing:
            logger.error(f"  Missing: {', '.join(missing)}")
            logger.info(f"  Install dependencies: pip install -r {requirements_file}")
            return False

        logger.info("  ✓ All dependencies installed")
        return True

    except Exception as e:
        logger.error(f"  Dependency check failed: {str(e)}")
        return False


def load_environment():
    """Load environment variables from .env file."""
    logger.info("Loading environment variables...")

    env_file = Path(__file__).parent.parent / ".env"

    if not env_file.exists():
        logger.error(f"  {env_file} not found")
        return False

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

        logger.info("  ✓ Environment variables loaded")
        return True
    except Exception as e:
        logger.error(f"  Failed to load environment: {str(e)}")
        return False


def create_directories():
    """Create required project directories."""
    logger.info("Creating directories...")

    from config import settings

    try:
        dirs = [settings.knowledge_dir, settings.processed_dir, settings.failed_dir]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("  ✓ Directories created")
        return True
    except Exception as e:
        logger.error(f"  Failed to create directories: {str(e)}")
        return False


def setup_database():
    """Run database setup script."""
    logger.info("Setting up database...")

    script_file = Path(__file__).parent / "setup_db.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_file)],
            capture_output=False,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("  ✓ Database setup completed")
            return True
        else:
            logger.warning("  ⚠️  Database setup failed or requires manual intervention")
            return False

    except subprocess.TimeoutExpired:
        logger.error("  Database setup timed out")
        return False
    except Exception as e:
        logger.error(f"  Failed to run database setup: {str(e)}")
        return False


def validate_config():
    """Validate final configuration."""
    logger.info("Validating configuration...")

    try:
        from config import settings

        logger.info(f"  ✓ Config loaded: {settings}")
        return True
    except Exception as e:
        logger.error(f"  Configuration validation failed: {str(e)}")
        return False


def main():
    """Main setup function."""
    logger.info("=" * 70)
    logger.info("🚀 Project Setup Script - Langchain Agent RAG System")
    logger.info("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Load Environment", load_environment),
        ("Create Directories", create_directories),
        ("Database Setup", setup_database),
        ("Validate Config", validate_config),
    ]

    results = []

    for check_name, check_func in checks:
        logger.info(f"\n[{len(results)+1}/{len(checks)}] {check_name}")
        logger.info("-" * 70)

        try:
            success = check_func()
            results.append((check_name, success))

            if not success:
                logger.warning(f"⚠️  {check_name} failed - some functionality may not work")
                # Continue with other checks instead of failing completely
        except Exception as e:
            logger.error(f"❌ {check_name} error: {str(e)}", exc_info=True)
            results.append((check_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📋 Setup Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for check_name, success in results:
        status = "✅" if success else "⚠️ "
        logger.info(f"{status} {check_name}")

    logger.info(f"\nPassed: {passed}/{total}")

    if passed == total:
        logger.info("\n✅ Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. python main.py          # Start the ingestion system")
        logger.info("  2. Drop files in /knowledge/raw_docs/")
        logger.info("  3. Files will be auto-processed every 5 minutes")
        logger.info("\nOr for manual testing:")
        logger.info("  python -c 'from main import *; from config import settings'")
        return 0
    else:
        logger.warning("\n⚠️  Setup partially complete - some checks failed")
        logger.warning("     Review warnings above and fix as needed")
        logger.warning("\nYou can still try running the system, but some features may not work")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
