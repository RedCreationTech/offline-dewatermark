import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wm import logging_setup


class ConfiguredLevelTests(unittest.TestCase):
    def test_default_is_info(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(logging_setup, 'CONFIG_PATH', Path('/nonexistent/logging.json')):
            self.assertEqual(logging_setup.configured_level(), 'INFO')

    def test_config_file_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'logging.json'
            cfg.write_text(json.dumps({'level': 'debug'}), 'utf-8')
            with patch.object(logging_setup, 'CONFIG_PATH', cfg), \
                 patch.dict(os.environ, {}, clear=True):
                self.assertEqual(logging_setup.configured_level(), 'DEBUG')

    def test_env_overrides_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'logging.json'
            cfg.write_text(json.dumps({'level': 'ERROR'}), 'utf-8')
            with patch.object(logging_setup, 'CONFIG_PATH', cfg), \
                 patch.dict(os.environ, {'WM_LOG_LEVEL': 'warning'}):
                self.assertEqual(logging_setup.configured_level(), 'WARNING')

    def test_invalid_values_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'logging.json'
            cfg.write_text('{bad json', 'utf-8')
            with patch.object(logging_setup, 'CONFIG_PATH', cfg), \
                 patch.dict(os.environ, {'WM_LOG_LEVEL': 'NOPE'}):
                self.assertEqual(logging_setup.configured_level(), 'INFO')


class SetupLoggingTests(unittest.TestCase):
    def test_setup_writes_file_and_applies_level(self):
        logger = logging.getLogger('wm')
        old_handlers = logger.handlers[:]
        old_level = logger.level
        logger.handlers = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_file = Path(tmp) / 'wm.log'
                cfg = Path(tmp) / 'logging.json'
                cfg.write_text(json.dumps({'level': 'DEBUG'}), 'utf-8')
                with patch.object(logging_setup, 'CONFIG_PATH', cfg), \
                     patch.object(logging_setup, 'LOG_DIR', Path(tmp)), \
                     patch.object(logging_setup, 'LOG_FILE', log_file), \
                     patch.dict(os.environ, {}, clear=True):
                    returned = logging_setup.setup_logging()
                    self.assertEqual(returned, log_file)
                    logging.getLogger('wm.test').debug('调试行')
                    for handler in logger.handlers:
                        handler.flush()
                    text = log_file.read_text('utf-8')
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
                self.assertIn('调试行', text)
                self.assertIn('DEBUG', text)
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)

    def test_apply_level_rejects_unknown(self):
        self.assertEqual(logging_setup.apply_level('chatty'), 'INFO')

    def test_persist_level_writes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'logging.json'
            with patch.object(logging_setup, 'CONFIG_PATH', cfg):
                logging_setup.persist_level('DEBUG')
                self.assertEqual(json.loads(cfg.read_text('utf-8'))['level'], 'DEBUG')
            logging_setup.persist_level('NOT_A_LEVEL')
            self.assertEqual(json.loads(cfg.read_text('utf-8'))['level'], 'DEBUG')


class GuiLogControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wm.app import make_app
        cls.app = make_app()

    def test_combo_changes_level_and_persists(self):
        from wm.app import MainWindow
        logger = logging.getLogger('wm')
        old_level = logger.level
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'logging.json'
            cfg.write_text(json.dumps({'level': 'INFO'}), 'utf-8')
            with patch.object(logging_setup, 'CONFIG_PATH', cfg):
                w = MainWindow()
                try:
                    w.log_level_box.setCurrentText('DEBUG')
                    self.assertEqual(logger.level, logging.DEBUG)
                    self.assertEqual(json.loads(cfg.read_text('utf-8'))['level'], 'DEBUG')
                    # 模拟启动后直接改配置文件：监听回调应把界面和级别同步过去
                    cfg.write_text(json.dumps({'level': 'ERROR'}), 'utf-8')
                    w._reload_log_level(str(cfg))
                    self.assertEqual(logger.level, logging.ERROR)
                    self.assertEqual(w.log_level_box.currentText(), 'ERROR')
                finally:
                    w.close()
        logger.setLevel(old_level)


if __name__ == '__main__':
    unittest.main()
