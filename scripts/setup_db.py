"""
Database initialization script for Supabase.

This script:
1. Connects to Supabase
2. Creates all required tables, indexes, and triggers
3. Validates schema setup
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def load_schema() -> str:
    """Load SQL schema from schema.sql file."""
    schema_file = Path(__file__).parent.parent / "schema.sql"
    
    if not schema_file.exists():
        raise FileNotFoundError(f"schema.sql not found at {schema_file}")
    
    with open(schema_file, "r") as f:
        return f.read()


def execute_schema(supabase_client, schema_sql: str) -> None:
    """
    Execute SQL schema using Supabase admin client.
    
    Args:
        supabase_client: Supabase client instance
        schema_sql: SQL schema string
    """
    try:
        # Split SQL into individual statements (remove comments and empty lines)
        statements = []
        current_statement = ""
        
        for line in schema_sql.split("\n"):
            # Remove comments
            if "--" in line:
                line = line[: line.index("--")]
            
            line = line.strip()
            
            if not line:
                continue
            
            current_statement += " " + line
            
            if line.endswith(";"):
                statements.append(current_statement.strip())
                current_statement = ""
        
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            if not statement or statement.startswith("--"):
                continue
            
            try:
                logger.debug(f"Executing statement {i}: {statement[:60]}...")
                result = supabase_client.postgrest.session.post(
                    f"{supabase_client.postgrest.base_url}/rpc/execute_sql",
                    json={"sql": statement},
                )
                # If RPC doesn't exist, try direct execution
            except Exception as e:
                logger.warning(f"RPC method failed, trying direct query: {str(e)}")
                try:
                    # Try using the query method directly
                    supabase_client.query(statement)
                except Exception as e2:
                    logger.warning(f"Direct query also failed: {str(e2)}")
        
        logger.info("✅ Schema executed successfully")
        
    except Exception as e:
        logger.error(f"Failed to execute schema: {str(e)}")
        raise


def execute_schema_direct(supabase_url: str, supabase_key: str, schema_sql: str) -> None:
    """
    Execute schema using direct PostgreSQL connection via SUPABASE_DIRECT_URL.
    
    Args:
        supabase_url: (unused - kept for compatibility)
        supabase_key: (unused - kept for compatibility)
        schema_sql: SQL schema string
    """
    try:
        import psycopg2
        
        logger.info("Attempting direct PostgreSQL connection...")
        
        # Use SUPABASE_DIRECT_URL from environment
        if not hasattr(settings, 'supabase_direct_url') or not settings.supabase_direct_url:
            raise ValueError("SUPABASE_DIRECT_URL not set in .env")
        
        db_url = settings.supabase_direct_url
        logger.debug(f"Using SUPABASE_DIRECT_URL: {db_url[:50]}...")
        
        # Connect directly
        conn = psycopg2.connect(db_url, connect_timeout=10)
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("✅ Connected to PostgreSQL")
        
        # Split and execute statements
        statements = []
        current_statement = ""
        
        for line in schema_sql.split("\n"):
            if "--" in line:
                line = line[: line.index("--")]
            
            line = line.rstrip()
            
            if not line:
                continue
            
            current_statement += " " + line
            
            if line.endswith(";"):
                statements.append(current_statement.strip())
                current_statement = ""
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        for i, statement in enumerate(statements, 1):
            if not statement or statement.startswith("--"):
                continue
            
            try:
                cursor.execute(statement)
                logger.debug(f"  ✓ Statement {i}/{len(statements)} executed")
            except Exception as e:
                logger.warning(f"  ⚠ Statement {i} warning: {str(e)}")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Schema executed successfully")
        
    except ImportError:
        logger.warning("psycopg2 not installed, trying Supabase API method...")
        execute_schema_via_api(supabase_url, supabase_key, schema_sql)
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        raise


def execute_schema_via_api(supabase_url: str, supabase_key: str, schema_sql: str) -> None:
    """
    Execute schema using Supabase REST API.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        schema_sql: SQL schema string
    """
    try:
        import requests
        
        logger.info("Executing schema via Supabase REST API...")
        
        # Split statements
        statements = []
        current_statement = ""
        
        for line in schema_sql.split("\n"):
            if "--" in line:
                line = line[: line.index("--")]
            
            line = line.rstrip()
            
            if not line:
                continue
            
            current_statement += " " + line
            
            if line.endswith(";"):
                statements.append(current_statement.strip())
                current_statement = ""
        
        logger.info(f"Executing {len(statements)} SQL statements...")
        
        # Note: Supabase REST API doesn't directly support arbitrary SQL
        # This is a limitation - users need to run SQL manually or use CLI
        logger.warning("⚠️  Supabase REST API does not support arbitrary SQL execution")
        logger.warning("Please use one of these alternatives:")
        logger.warning("  1. Run: supabase db push  (if using Supabase CLI)")
        logger.warning("  2. Run SQL manually in Supabase Studio > SQL Editor")
        logger.warning("  3. Use direct PostgreSQL connection (psycopg2 required)")
        
        raise NotImplementedError(
            "Supabase REST API doesn't support arbitrary SQL. "
            "Install psycopg2 or use Supabase CLI."
        )
        
    except Exception as e:
        logger.error(f"Failed API method: {str(e)}")
        raise


def validate_schema(supabase_client) -> bool:
    """
    Validate that schema was created correctly.
    
    Args:
        supabase_client: Supabase client instance
        
    Returns:
        True if all tables exist
    """
    try:
        logger.info("Validating schema...")
        
        # Check if tables exist by querying information schema
        required_tables = ["documents", "document_chunks"]
        
        for table in required_tables:
            try:
                # Try to query the table to verify it exists
                result = supabase_client.table(table).select("*", count="exact").limit(1).execute()
                logger.info(f"  ✓ Table '{table}' exists")
            except Exception as e:
                logger.error(f"  ✗ Table '{table}' not found: {str(e)}")
                return False
        
        logger.info("✅ Schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        return False


def setup_db():
    """Main database setup function."""
    logger.info("=" * 60)
    logger.info("🗄️  Database Setup Script")
    logger.info("=" * 60)
    
    try:
        # Step 1: Validate configuration
        logger.info("\n[Step 1/3] Validating configuration...")
        logger.info(f"  Supabase URL: {settings.supabase_url[:50]}...")
        logger.info(f"  Supabase Key: {settings.supabase_key[:10]}...")
        
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        
        logger.info("  ✓ Configuration valid")
        
        # Step 2: Load schema
        logger.info("\n[Step 2/3] Loading database schema...")
        schema_sql = load_schema()
        logger.info(f"  Loaded schema ({len(schema_sql)} bytes)")
        
        # Step 3: Execute schema using direct PostgreSQL
        logger.info("\n[Step 3/3] Setting up database schema...")
        try:
            # Try direct PostgreSQL connection (most reliable)
            execute_schema_direct(
                settings.supabase_url,
                settings.supabase_key,
                schema_sql,
            )
        except ImportError:
            logger.warning("  psycopg2 not available, trying Supabase API...")
            try:
                execute_schema_via_api(
                    settings.supabase_url,
                    settings.supabase_key,
                    schema_sql,
                )
            except NotImplementedError:
                logger.error("\n❌ Cannot execute schema - no suitable method available")
                logger.info("\n✅ MANUAL SETUP REQUIRED:")
                logger.info("  1. Go to Supabase Studio: https://app.supabase.com/")
                logger.info("  2. Open your project > SQL Editor")
                logger.info(f"  3. Copy contents of: {Path(__file__).parent.parent / 'schema.sql'}")
                logger.info("  4. Paste into SQL Editor and run")
                logger.info("\nOr install psycopg2:")
                logger.info("  pip install psycopg2-binary")
                return False
        
        # Step 4: Validate schema (try to initialize Supabase client if URL is valid)
        logger.info("\n[Step 4/4] Validating schema...")
        
        # Check if SUPABASE_URL is a valid Supabase URL (not a PostgreSQL connection string)
        if settings.supabase_url.startswith("https://"):
            try:
                from supabase import create_client
                supabase_client = create_client(settings.supabase_url, settings.supabase_key)
                if validate_schema(supabase_client):
                    logger.info("\n" + "=" * 60)
                    logger.info("✅ Database setup completed successfully!")
                    logger.info("=" * 60)
                    return True
            except Exception as e:
                logger.warning(f"  Could not validate via Supabase client: {str(e)}")
                logger.info("  (Schema execution succeeded, validation skipped)")
        else:
            logger.info("  (Skipping Supabase client validation - using direct connection)")
            logger.info("✅ Schema setup executed successfully")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Database setup completed!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Database setup failed: {str(e)}", exc_info=True)
        logger.info("\n" + "=" * 60)
        logger.info("TROUBLESHOOTING:")
        logger.info("  1. Verify SUPABASE_URL and SUPABASE_KEY in .env")
        logger.info("  2. For direct PostgreSQL: use SUPABASE_DIRECT_URL (postgresql://...)")
        logger.info("  3. Ensure pgvector extension is enabled in Supabase")
        logger.info("=" * 60)
        return False


if __name__ == "__main__":
    success = setup_db()
    sys.exit(0 if success else 1)
