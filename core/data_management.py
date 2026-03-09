"""
Day 18: Data Management System for Financial Distress Early Warning System
Comprehensive database integration, migrations, backup, archiving, and import/export
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import pickle
import csv
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class DataRetentionPolicy(Enum):
    """Data retention duration options"""
    TEMP = 7  # 7 days
    SHORT_TERM = 30  # 30 days
    MEDIUM_TERM = 90  # 90 days
    LONG_TERM = 365  # 1 year
    PERMANENT = None  # Keep forever


class DataValidationRule(Enum):
    """Data validation rule types"""
    REQUIRED = "required"
    NUMERIC = "numeric"
    POSITIVE = "positive"
    RANGE = "range"
    DATE = "date"
    EMAIL = "email"
    UNIQUE = "unique"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = 'instance/app.db'
    backup_dir: str = 'backups'
    archive_dir: str = 'archives'
    max_connections: int = 10
    timeout: int = 30
    isolation_level: str = 'DEFERRED'


class DatabaseMigration:
    """Database schema migration"""

    def __init__(self, version: int, description: str, migration_sql: List[str]):
        self.version = version
        self.description = description
        self.migration_sql = migration_sql
        self.applied_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'description': self.description,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None
        }


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._ensure_paths()
        self.migrations: List[DatabaseMigration] = []
        self._initialize_database()

    def _ensure_paths(self) -> None:
        """Ensure required directories exist"""
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.archive_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_database(self) -> None:
        """Initialize database with schema"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create migrations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY,
                    version INTEGER UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create data audit table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_audit_log (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    record_id INTEGER,
                    old_data TEXT,
                    new_data TEXT,
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create data validation rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_rules (
                    id INTEGER PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    rule_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(
            self.config.db_path,
            timeout=self.config.timeout,
            isolation_level=self.config.isolation_level
        )
        conn.row_factory = sqlite3.Row
        return conn

    def register_migration(self, migration: DatabaseMigration) -> None:
        """Register a database migration"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)

    def apply_migrations(self) -> List[DatabaseMigration]:
        """Apply pending migrations"""
        applied = []
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            for migration in self.migrations:
                # Check if already applied
                cursor.execute(
                    'SELECT 1 FROM schema_migrations WHERE version = ?',
                    (migration.version,)
                )

                if cursor.fetchone():
                    logger.info(f"Migration {migration.version} already applied")
                    continue

                # Apply migration
                for sql in migration.migration_sql:
                    cursor.execute(sql)

                # Record migration
                cursor.execute(
                    'INSERT INTO schema_migrations (version, description) VALUES (?, ?)',
                    (migration.version, migration.description)
                )

                migration.applied_at = datetime.now(timezone.utc)
                applied.append(migration)
                logger.info(f"Applied migration {migration.version}: {migration.description}")

            conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Error applying migrations: {str(e)}")
            raise
        finally:
            conn.close()

        return applied

    def get_migration_status(self) -> List[Dict]:
        """Get status of all migrations"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM schema_migrations ORDER BY version')
            migrations = cursor.fetchall()
            return [dict(m) for m in migrations]
        finally:
            conn.close()


class DataValidator:
    """Validates data against configured rules"""

    def __init__(self):
        self.rules: Dict[str, List[Dict]] = {}

    def add_rule(self, table_name: str, column_name: str, rule_type: DataValidationRule, config: Optional[Dict] = None) -> None:
        """Add validation rule"""
        key = f"{table_name}.{column_name}"

        if key not in self.rules:
            self.rules[key] = []

        self.rules[key].append({
            'type': rule_type.value,
            'config': config or {}
        })

    def validate_value(self, table_name: str, column_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single value"""
        key = f"{table_name}.{column_name}"

        if key not in self.rules:
            return True, None

        for rule in self.rules[key]:
            is_valid, error = self._check_rule(rule['type'], value, rule.get('config', {}))
            if not is_valid:
                return False, error

        return True, None

    def validate_record(self, table_name: str, record: Dict) -> Tuple[bool, List[str]]:
        """Validate entire record"""
        errors = []

        for column_name, value in record.items():
            is_valid, error = self.validate_value(table_name, column_name, value)
            if not is_valid:
                errors.append(f"{column_name}: {error}")

        return len(errors) == 0, errors

    def _check_rule(self, rule_type: str, value: Any, config: Dict) -> Tuple[bool, Optional[str]]:
        """Check individual validation rule"""
        if rule_type == DataValidationRule.REQUIRED.value:
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return False, "Value is required"

        elif rule_type == DataValidationRule.NUMERIC.value:
            try:
                float(value)
            except (ValueError, TypeError):
                return False, "Value must be numeric"

        elif rule_type == DataValidationRule.POSITIVE.value:
            try:
                if float(value) <= 0:
                    return False, "Value must be positive"
            except (ValueError, TypeError):
                return False, "Value must be numeric"

        elif rule_type == DataValidationRule.RANGE.value:
            try:
                min_val = config.get('min')
                max_val = config.get('max')
                val = float(value)

                if min_val is not None and val < min_val:
                    return False, f"Value must be >= {min_val}"
                if max_val is not None and val > max_val:
                    return False, f"Value must be <= {max_val}"

            except (ValueError, TypeError):
                return False, "Value must be numeric"

        elif rule_type == DataValidationRule.DATE.value:
            try:
                datetime.fromisoformat(str(value))
            except (ValueError, TypeError):
                return False, "Value must be valid ISO format date"

        elif rule_type == DataValidationRule.EMAIL.value:
            if '@' not in str(value) or '.' not in str(value):
                return False, "Value must be valid email"

        return True, None


class DataBackupManager:
    """Manages database backups"""

    def __init__(self, db_path: str, backup_dir: str = 'backups'):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create database backup"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backup_dir / f"{backup_name}.db"

        try:
            conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(str(backup_path))

            conn.backup(backup_conn)
            backup_conn.close()
            conn.close()

            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise

    def restore_backup(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            backup_path = Path(backup_path)

            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")

            backup_conn = sqlite3.connect(str(backup_path))
            conn = sqlite3.connect(self.db_path)

            backup_conn.backup(conn)
            conn.close()
            backup_conn.close()

            logger.info(f"Database restored from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            return False

    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []

        for backup_file in self.backup_dir.glob("*.db"):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

    def delete_old_backups(self, keep_count: int = 10) -> int:
        """Delete old backups, keeping only recent ones"""
        backups = self.list_backups()
        deleted = 0

        for backup in backups[keep_count:]:
            try:
                Path(backup['path']).unlink()
                deleted += 1
                logger.info(f"Deleted old backup: {backup['name']}")
            except Exception as e:
                logger.error(f"Error deleting backup: {str(e)}")

        return deleted


class DataArchiver:
    """Archives and retrieves historical data"""

    def __init__(self, archive_dir: str = 'archives'):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_data(self, table_name: str, data: List[Dict], archive_date: Optional[datetime] = None) -> str:
        """Archive data to file"""
        if archive_date is None:
            archive_date = datetime.now(timezone.utc)

        archive_name = f"{table_name}_{archive_date.strftime('%Y%m%d')}.pkl"
        archive_path = self.archive_dir / archive_name

        try:
            with open(archive_path, 'wb') as f:
                pickle.dump({
                    'table': table_name,
                    'date': archive_date.isoformat(),
                    'record_count': len(data),
                    'data': data
                }, f)

            logger.info(f"Data archived: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.error(f"Error archiving data: {str(e)}")
            raise

    def retrieve_archive(self, archive_path: str) -> Dict:
        """Retrieve archived data"""
        try:
            with open(archive_path, 'rb') as f:
                archive_data = pickle.load(f)

            logger.info(f"Archive retrieved: {archive_path}")
            return archive_data

        except Exception as e:
            logger.error(f"Error retrieving archive: {str(e)}")
            raise

    def list_archives(self, table_name: Optional[str] = None) -> List[Dict]:
        """List all archives"""
        archives = []

        pattern = f"{table_name}_*.pkl" if table_name else "*.pkl"

        for archive_file in self.archive_dir.glob(pattern):
            stat = archive_file.stat()
            archives.append({
                'name': archive_file.name,
                'path': str(archive_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return sorted(archives, key=lambda x: x['created_at'], reverse=True)

    def cleanup_old_archives(self, policy: DataRetentionPolicy) -> int:
        """Delete archives older than retention policy"""
        if policy.value is None:
            return 0

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.value)
        deleted = 0

        for archive_file in self.archive_dir.glob("*.pkl"):
            if datetime.fromtimestamp(archive_file.stat().st_mtime).replace(tzinfo=timezone.utc) < cutoff_date:
                try:
                    archive_file.unlink()
                    deleted += 1
                    logger.info(f"Deleted old archive: {archive_file.name}")
                except Exception as e:
                    logger.error(f"Error deleting archive: {str(e)}")

        return deleted


class DataImporter:
    """Imports data from various formats"""

    def __init__(self, validator: Optional[DataValidator] = None):
        self.validator = validator or DataValidator()

    def import_from_csv(self, file_path: str, table_name: str, validate: bool = True) -> Tuple[int, List[str]]:
        """Import data from CSV file"""
        imported = 0
        errors = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row_num, row in enumerate(reader, 1):
                    if validate:
                        is_valid, validation_errors = self.validator.validate_record(table_name, row)
                        if not is_valid:
                            errors.append(f"Row {row_num}: {', '.join(validation_errors)}")
                            continue

                    imported += 1

            logger.info(f"Imported {imported} records from {file_path}")
            return imported, errors

        except Exception as e:
            logger.error(f"Error importing CSV: {str(e)}")
            raise

    def import_from_json(self, file_path: str, table_name: str, validate: bool = True) -> Tuple[int, List[str]]:
        """Import data from JSON file"""
        imported = 0
        errors = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if not isinstance(data, list):
                    data = [data]

                for row_num, record in enumerate(data, 1):
                    if validate:
                        is_valid, validation_errors = self.validator.validate_record(table_name, record)
                        if not is_valid:
                            errors.append(f"Record {row_num}: {', '.join(validation_errors)}")
                            continue

                    imported += 1

            logger.info(f"Imported {imported} records from {file_path}")
            return imported, errors

        except Exception as e:
            logger.error(f"Error importing JSON: {str(e)}")
            raise


class DataExporter:
    """Exports data in various formats"""

    def export_to_csv(self, data: List[Dict], output_path: str) -> str:
        """Export data to CSV"""
        try:
            if not data:
                logger.warning("No data to export")
                return output_path

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Data exported to CSV: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            raise

    def export_to_json(self, data: List[Dict], output_path: str) -> str:
        """Export data to JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Data exported to JSON: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            raise

    def export_to_jsonl(self, data: List[Dict], output_path: str) -> str:
        """Export data to JSONL (one JSON per line)"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record, default=str) + '\n')

            logger.info(f"Data exported to JSONL: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting JSONL: {str(e)}")
            raise


class DataManagementEngine:
    """Main orchestrator for all data management operations"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.db_manager = DatabaseManager(self.config)
        self.validator = DataValidator()
        self.backup_manager = DataBackupManager(self.config.db_path, self.config.backup_dir)
        self.archiver = DataArchiver(self.config.archive_dir)
        self.importer = DataImporter(self.validator)
        self.exporter = DataExporter()

    def get_database_status(self) -> Dict:
        """Get comprehensive database status"""
        conn = None
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            result = cursor.fetchone()
            db_size = result[0] if result else 0

            return {
                'database_path': self.config.db_path,
                'size_mb': db_size / (1024 * 1024),
                'migrations': self.db_manager.get_migration_status(),
                'backups_available': len(self.backup_manager.list_backups()),
                'archives_available': len(self.archiver.list_archives())
            }

        except Exception as e:
            logger.error(f"Error getting database status: {str(e)}")
            return {'error': str(e)}
        finally:
            if conn is not None:
                conn.close()

    def add_validation_rule(self, table_name: str, column_name: str, rule_type: str, config: Optional[Dict] = None) -> None:
        """Add data validation rule"""
        try:
            rule_enum = DataValidationRule[rule_type.upper()]
            self.validator.add_rule(table_name, column_name, rule_enum, config)
            logger.info(f"Added validation rule: {table_name}.{column_name} = {rule_type}")
        except KeyError:
            logger.error(f"Unknown validation rule: {rule_type}")

    def perform_backup(self, backup_name: Optional[str] = None) -> str:
        """Perform database backup"""
        return self.backup_manager.create_backup(backup_name)

    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore database from backup"""
        return self.backup_manager.restore_backup(backup_path)

    def list_backups(self) -> List[Dict]:
        """List all backups"""
        return self.backup_manager.list_backups()

    def archive_data(self, table_name: str, data: List[Dict]) -> str:
        """Archive table data"""
        return self.archiver.archive_data(table_name, data)

    def list_archives(self, table_name: Optional[str] = None) -> List[Dict]:
        """List all archives"""
        return self.archiver.list_archives(table_name)

    def cleanup_old_data(self, retention_policy: DataRetentionPolicy) -> int:
        """Clean up old archived data"""
        return self.archiver.cleanup_old_archives(retention_policy)

    def import_data(self, file_path: str, table_name: str, file_format: str = 'csv', validate: bool = True) -> Tuple[int, List[str]]:
        """Import data from file"""
        if file_format.lower() == 'csv':
            return self.importer.import_from_csv(file_path, table_name, validate)
        elif file_format.lower() == 'json':
            return self.importer.import_from_json(file_path, table_name, validate)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def export_data(self, data: List[Dict], output_path: str, file_format: str = 'json') -> str:
        """Export data to file"""
        if file_format.lower() == 'csv':
            return self.exporter.export_to_csv(data, output_path)
        elif file_format.lower() == 'json':
            return self.exporter.export_to_json(data, output_path)
        elif file_format.lower() == 'jsonl':
            return self.exporter.export_to_jsonl(data, output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def register_migration(self, version: int, description: str, migration_sql: List[str]) -> None:
        """Register database migration"""
        migration = DatabaseMigration(version, description, migration_sql)
        self.db_manager.register_migration(migration)

    def apply_migrations(self) -> List[DatabaseMigration]:
        """Apply all pending migrations"""
        return self.db_manager.apply_migrations()

    def get_migration_status(self) -> List[Dict]:
        """Get migration status"""
        return self.db_manager.get_migration_status()
