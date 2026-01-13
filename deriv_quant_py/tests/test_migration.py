import unittest
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from deriv_quant_py.database import init_db

# Mock Base for Old Schema
OldBase = declarative_base()
class OldStrategyParams(OldBase):
    __tablename__ = 'strategy_params'
    symbol = Column(String, primary_key=True)
    rsi_period = Column(Integer)
    # Missing new columns

class TestDatabaseMigration(unittest.TestCase):
    def setUp(self):
        self.db_path = "sqlite:///test_migration.db"
        self.file_path = "test_migration.db"
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_migration_adds_columns(self):
        # 1. Create DB with OLD schema
        engine = create_engine(self.db_path)
        OldBase.metadata.create_all(engine)

        # Verify columns are missing
        from sqlalchemy import inspect
        inspector = inspect(engine)
        cols = [c['name'] for c in inspector.get_columns('strategy_params')]
        self.assertNotIn('optimal_duration', cols)
        self.assertNotIn('rsi_vol_window', cols)

        # 2. Run init_db (should migrate)
        session_factory = init_db(self.db_path)

        # 3. Verify columns exist
        inspector = inspect(engine)
        new_cols = [c['name'] for c in inspector.get_columns('strategy_params')]

        self.assertIn('optimal_duration', new_cols)
        self.assertIn('rsi_vol_window', new_cols)
        self.assertIn('details', new_cols)

if __name__ == '__main__':
    unittest.main()
