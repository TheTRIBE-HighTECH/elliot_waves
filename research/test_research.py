"""Hermetic tests for GOX annotations, collector structure, and causality."""
import unittest, datetime as dt, json, tempfile, pathlib
from unittest.mock import patch
from gox_annotations import convert
from atlas_snapshots import collect

class GoxTests(unittest.TestCase):
    def test_capture(self):
        obj={
          "symbol":"BTCUSDT","timeframe":"1h","annotator":"analyst-1",
          "direction":"bull","published_at":"2025-05-03T10:00:00Z",
          "waves":[{"wave":"3","degree":"Minor","start_date":"2025-05-01T00:00:00Z",
                    "end_date":"2025-05-02T00:00:00Z","end_price":100000,
                    "confirmed_at":"2025-05-03T00:00:00Z"}]}
        x=convert(obj)
        self.assertEqual(x[0]["wave"],"3")
        self.assertEqual(x[0]["extreme_price"],100000)
        self.assertEqual(x[0]["label_status"],"human_unverified")
        self.assertLess(x[0]["extreme_time"],x[0]["confirmation_time"])
    def test_reject_time_travel(self):
        obj={"asset":"BTCUSDT","timeframe":"1d","annotator":"x","direction":"bull",
             "waves":[{"wave":"1","degree":"Minor","end_date":"2025-01-10T00:00:00Z",
                       "end_price":110,"confirmed_at":"2025-01-09T00:00:00Z"}]}
        with self.assertRaises(ValueError):convert(obj)
    def test_reject_missing_provenance(self):
        with self.assertRaises(ValueError):convert({"asset":"BTCUSDT","waves":[]})
    def test_reject_unrecognized_wave(self):
        obj={"asset":"BTCUSDT","timeframe":"1d","annotator":"x",
             "waves":[{"wave":"nonsense","degree":"Minor"}]}
        self.assertEqual(convert(obj),[])

class AtlasTests(unittest.TestCase):
    def test_capture_and_metadata(self):
        index={"updated":"2026-11-23","markets":[{"slug":"btc-usd","symbol":"BTC-USD"}]}
        market={"symbol":"BTC-USD","asof":"2026-11-23","levels":[{"wave":"3"}]}
        def read(path,timeout=30):
            return index if path.endswith("index.json") else market
        with tempfile.TemporaryDirectory() as tmp:
            with patch("atlas_snapshots.read_json",side_effect=read):
                with patch("atlas_snapshots.time.sleep",return_value=None):
                    x=collect(tmp,1,.5)
            self.assertEqual(x["market_count"],1)
            self.assertTrue(x["snapshots"][0]["sha256"])
            fn=list(pathlib.Path(tmp).rglob("btc-usd.json"))
            self.assertEqual(len(fn),1)
            self.assertEqual(json.loads(fn[0].read_text())["symbol"],"BTC-USD")
    def test_empty_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("atlas_snapshots.read_json",return_value={"markets":[]}):
                with self.assertRaises(ValueError):collect(tmp,1,.5)

if __name__=="__main__":unittest.main()
