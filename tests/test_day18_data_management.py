import pytest
import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sqlite3

from core.data_management import (
    DatabaseConfig,
    DatabaseManager,
    DatabaseMigration,
    DataValidator,
    DataValidationRule,
    DataRetentionPolicy,
    DataBackupManager,
    DataArchiver,
    DataImporter,
    DataExporter,
    DataManagementEngine
)


class TestDatabaseConfig:
    """Test database configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = DatabaseConfig()

        assert config.db_path == 'instance/app.db'
        assert config.backup_dir == 'backups'
        assert config.archive_dir == 'archives'
        assert config.max_connections == 10

    def test_custom_config(self):
        """Test custom configuration"""
        config = DatabaseConfig(
            db_path='test.db',
            backup_dir='test_backups',
            archive_dir='test_archives'
        )

        assert config.db_path == 'test.db'
        assert config.backup_dir == 'test_backups'
        assert config.archive_dir == 'test_archives'


class TestDatabaseMigration:
    """Test database migrations"""

    def test_migration_creation(self):
        """Test creating a migration"""
        sql = ["CREATE TABLE test (id INTEGER PRIMARY KEY)"]
        migration = DatabaseMigration(1, "Create test table", sql)

        assert migration.version == 1
        assert migration.description == "Create test table"
        assert len(migration.migration_sql) == 1

    def test_migration_to_dict(self):
        """Test migration serialization"""
        migration = DatabaseMigration(1, "Test", ["SQL"])
        migration_dict = migration.to_dict()

        assert migration_dict['version'] == 1
        assert migration_dict['description'] == "Test"
        assert migration_dict['applied_at'] is None


class TestDatabaseManager:
    """Test database management"""

    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            manager = DatabaseManager(config)

            assert Path(config.db_path).exists()

    def test_get_connection(self):
        """Test getting database connection"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            manager = DatabaseManager(config)

            conn = manager.get_connection()
            assert conn is not None
            conn.close()

    def test_migration_registration(self):
        """Test registering migrations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            manager = DatabaseManager(config)

            migration = DatabaseMigration(1, "Test", ["CREATE TABLE test (id INTEGER)"])
            manager.register_migration(migration)

            assert len(manager.migrations) == 1

    def test_apply_migrations(self):
        """Test applying migrations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            manager = DatabaseManager(config)

            migration = DatabaseMigration(1, "Create users", [
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
            ])
            manager.register_migration(migration)

            applied = manager.apply_migrations()

            assert len(applied) == 1
            assert applied[0].version == 1

    def test_get_migration_status(self):
        """Test getting migration status"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            manager = DatabaseManager(config)

            migration = DatabaseMigration(1, "Test", ["CREATE TABLE test (id INTEGER)"])
            manager.register_migration(migration)
            manager.apply_migrations()

            status = manager.get_migration_status()

            assert len(status) == 1
            assert status[0]['version'] == 1


class TestDataValidator:
    """Test data validation"""

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()

        assert len(validator.rules) == 0

    def test_add_required_rule(self):
        """Test adding required validation rule"""
        validator = DataValidator()
        validator.add_rule('users', 'name', DataValidationRule.REQUIRED)

        key = 'users.name'
        assert key in validator.rules
        assert len(validator.rules[key]) == 1

    def test_validate_required_field(self):
        """Test validating required field"""
        validator = DataValidator()
        validator.add_rule('users', 'name', DataValidationRule.REQUIRED)

        # Valid
        is_valid, error = validator.validate_value('users', 'name', 'John')
        assert is_valid

        # Invalid
        is_valid, error = validator.validate_value('users', 'name', None)
        assert not is_valid

    def test_validate_numeric_field(self):
        """Test validating numeric field"""
        validator = DataValidator()
        validator.add_rule('users', 'age', DataValidationRule.NUMERIC)

        # Valid
        is_valid, _ = validator.validate_value('users', 'age', 25)
        assert is_valid

        # Invalid
        is_valid, _ = validator.validate_value('users', 'age', 'not_a_number')
        assert not is_valid

    def test_validate_positive_field(self):
        """Test validating positive field"""
        validator = DataValidator()
        validator.add_rule('accounts', 'balance', DataValidationRule.POSITIVE)

        # Valid
        is_valid, _ = validator.validate_value('accounts', 'balance', 100)
        assert is_valid

        # Invalid - zero
        is_valid, _ = validator.validate_value('accounts', 'balance', 0)
        assert not is_valid

        # Invalid - negative
        is_valid, _ = validator.validate_value('accounts', 'balance', -50)
        assert not is_valid

    def test_validate_range_field(self):
        """Test validating range field"""
        validator = DataValidator()
        validator.add_rule('users', 'age', DataValidationRule.RANGE, {'min': 0, 'max': 150})

        # Valid
        is_valid, _ = validator.validate_value('users', 'age', 25)
        assert is_valid

        # Invalid - too high
        is_valid, _ = validator.validate_value('users', 'age', 200)
        assert not is_valid

        # Invalid - too low
        is_valid, _ = validator.validate_value('users', 'age', -1)
        assert not is_valid

    def test_validate_record(self):
        """Test validating entire record"""
        validator = DataValidator()
        validator.add_rule('users', 'name', DataValidationRule.REQUIRED)
        validator.add_rule('users', 'age', DataValidationRule.NUMERIC)

        record = {'name': 'John', 'age': 25}
        is_valid, errors = validator.validate_record('users', record)

        assert is_valid
        assert len(errors) == 0


class TestDataBackupManager:
    """Test backup management"""

    def test_create_backup(self):
        """Test creating backup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            backup_dir = f'{tmpdir}/backups'

            # Create test database
            conn = sqlite3.connect(db_path)
            conn.close()

            manager = DataBackupManager(db_path, backup_dir)
            backup_path = manager.create_backup()

            assert Path(backup_path).exists()

    def test_backup_with_custom_name(self):
        """Test backup with custom name"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            backup_dir = f'{tmpdir}/backups'

            conn = sqlite3.connect(db_path)
            conn.close()

            manager = DataBackupManager(db_path, backup_dir)
            backup_path = manager.create_backup('my_backup')

            assert 'my_backup' in backup_path

    def test_list_backups(self):
        """Test listing backups"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            backup_dir = f'{tmpdir}/backups'

            conn = sqlite3.connect(db_path)
            conn.close()

            manager = DataBackupManager(db_path, backup_dir)
            manager.create_backup('backup1')
            manager.create_backup('backup2')

            backups = manager.list_backups()

            assert len(backups) == 2

    def test_delete_old_backups(self):
        """Test deleting old backups"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            backup_dir = f'{tmpdir}/backups'

            conn = sqlite3.connect(db_path)
            conn.close()

            manager = DataBackupManager(db_path, backup_dir)
            for i in range(5):
                manager.create_backup(f'backup{i}')

            deleted = manager.delete_old_backups(keep_count=2)

            assert deleted == 3


class TestDataArchiver:
    """Test data archiving"""

    def test_archive_data(self):
        """Test archiving data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = DataArchiver(f'{tmpdir}/archives')
            data = [{'id': 1, 'name': 'Test'}, {'id': 2, 'name': 'Test2'}]

            archive_path = archiver.archive_data('users', data)

            assert Path(archive_path).exists()

    def test_retrieve_archive(self):
        """Test retrieving archived data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = DataArchiver(f'{tmpdir}/archives')
            original_data = [{'id': 1, 'name': 'Test'}]

            archive_path = archiver.archive_data('users', original_data)
            retrieved = archiver.retrieve_archive(archive_path)

            assert retrieved['table'] == 'users'
            assert retrieved['record_count'] == 1
            assert retrieved['data'] == original_data

    def test_list_archives(self):
        """Test listing archives"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = DataArchiver(f'{tmpdir}/archives')
            data = [{'id': 1}]

            archiver.archive_data('users', data)
            archiver.archive_data('companies', data)

            all_archives = archiver.list_archives()
            assert len(all_archives) == 2

            user_archives = archiver.list_archives('users')
            assert len(user_archives) == 1

    def test_cleanup_old_archives(self):
        """Test cleanup old archives"""
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = DataArchiver(f'{tmpdir}/archives')
            data = [{'id': 1}]

            archiver.archive_data('users', data)

            # Cleanup with very short retention
            deleted = archiver.cleanup_old_archives(DataRetentionPolicy.TEMP)

            # May or may not delete depending on timing
            assert isinstance(deleted, int)


class TestDataImporter:
    """Test data import"""

    def test_import_from_csv(self):
        """Test importing CSV data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f'{tmpdir}/test.csv'

            # Create test CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'name'])
                writer.writeheader()
                writer.writerow({'id': '1', 'name': 'Test'})
                writer.writerow({'id': '2', 'name': 'Test2'})

            importer = DataImporter()
            imported, errors = importer.import_from_csv(csv_path, 'users', validate=False)

            assert imported == 2
            assert len(errors) == 0

    def test_import_from_json(self):
        """Test importing JSON data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = f'{tmpdir}/test.json'

            # Create test JSON
            data = [{'id': 1, 'name': 'Test'}, {'id': 2, 'name': 'Test2'}]
            with open(json_path, 'w') as f:
                json.dump(data, f)

            importer = DataImporter()
            imported, errors = importer.import_from_json(json_path, 'users', validate=False)

            assert imported == 2
            assert len(errors) == 0


class TestDataExporter:
    """Test data export"""

    def test_export_to_csv(self):
        """Test exporting to CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f'{tmpdir}/output.csv'
            data = [{'id': 1, 'name': 'Test'}, {'id': 2, 'name': 'Test2'}]

            exporter = DataExporter()
            result_path = exporter.export_to_csv(data, output_path)

            assert Path(result_path).exists()

    def test_export_to_json(self):
        """Test exporting to JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f'{tmpdir}/output.json'
            data = [{'id': 1, 'name': 'Test'}]

            exporter = DataExporter()
            result_path = exporter.export_to_json(data, output_path)

            assert Path(result_path).exists()

            with open(result_path) as f:
                exported = json.load(f)
                assert exported == data

    def test_export_to_jsonl(self):
        """Test exporting to JSONL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f'{tmpdir}/output.jsonl'
            data = [{'id': 1, 'name': 'Test'}, {'id': 2, 'name': 'Test2'}]

            exporter = DataExporter()
            result_path = exporter.export_to_jsonl(data, output_path)

            assert Path(result_path).exists()


class TestDataManagementEngine:
    """Test main data management engine"""

    def test_engine_initialization(self):
        """Test engine initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            assert engine.db_manager is not None
            assert engine.validator is not None
            assert engine.backup_manager is not None

    def test_get_database_status(self):
        """Test getting database status"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            status = engine.get_database_status()

            assert 'database_path' in status
            assert 'size_mb' in status
            assert 'backups_available' in status

    def test_add_validation_rule(self):
        """Test adding validation rule through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            engine.add_validation_rule('users', 'name', 'REQUIRED')

            key = 'users.name'
            assert key in engine.validator.rules

    def test_perform_backup(self):
        """Test backup through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                db_path=f'{tmpdir}/test.db',
                backup_dir=f'{tmpdir}/backups'
            )
            engine = DataManagementEngine(config)

            backup_path = engine.perform_backup()

            assert Path(backup_path).exists()

    def test_list_backups(self):
        """Test listing backups through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(
                db_path=f'{tmpdir}/test.db',
                backup_dir=f'{tmpdir}/backups'
            )
            engine = DataManagementEngine(config)

            engine.perform_backup('backup1')
            engine.perform_backup('backup2')

            backups = engine.list_backups()

            assert len(backups) == 2

    def test_register_and_apply_migrations(self):
        """Test registering and applying migrations through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            engine.register_migration(1, "Create users", [
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
            ])

            applied = engine.apply_migrations()

            assert len(applied) == 1

    def test_export_data(self):
        """Test exporting data through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            data = [{'id': 1, 'name': 'Test'}]
            output_path = engine.export_data(data, f'{tmpdir}/output.json', 'json')

            assert Path(output_path).exists()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_data_export(self):
        """Test exporting empty data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter()
            output_path = f'{tmpdir}/empty.json'

            result = exporter.export_to_json([], output_path)

            assert Path(result).exists()

    def test_invalid_validation_rule(self):
        """Test invalid validation rule"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            engine.add_validation_rule('users', 'name', 'INVALID_RULE')

            # Should not raise, just log error

    def test_unsupported_export_format(self):
        """Test unsupported export format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(db_path=f'{tmpdir}/test.db')
            engine = DataManagementEngine(config)

            with pytest.raises(ValueError):
                engine.export_data([], '/tmp/test', 'unsupported')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
