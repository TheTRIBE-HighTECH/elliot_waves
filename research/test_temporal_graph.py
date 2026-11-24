"""Regression checks: real reply edges, causality, and graph neural fusion."""
import datetime as dt
import unittest
from unittest.mock import patch

from hn_reply_graph import flatten_story,harvest,save

def event_time(hour):
    return f"2025-06-15T{hour:02d}:15:00+00:00"
def tiny_thread():
    return {"id":100,"created_at":event_time(0),"title":"Bitcoin story",
            "children":[{"id":101,"created_at":event_time(1),"parent_id":100,
              "children":[{"id":102,"created_at":event_time(2),
                           "parent_id":101,"children":[]}]},
             {"id":103,"created_at":event_time(4),"parent_id":100,
              "children":[]}]}

class ReplySourceTests(unittest.TestCase):
    def test_actual_explicit_edges_are_preserved(self):
        start=dt.datetime(2025,6,15,tzinfo=dt.timezone.utc)
        end=start+dt.timedelta(days=1)
        rows=flatten_story(tiny_thread(),"BTCUSDT","bitcoin",start,end,100)
        self.assertEqual(len(rows),4)
        self.assertEqual({x["node_id"]:x["parent_id"] for x in rows},
                         {100:"",101:100,102:101,103:100})
        self.assertEqual([r["depth"] for r in rows],[0,1,2,1])
        self.assertNotIn("author",rows[0])
        self.assertNotIn("text",rows[0])

    def test_cannot_use_comments_after_end_of_collection_window(self):
        start=dt.datetime(2025,6,15,tzinfo=dt.timezone.utc)
        end=start+dt.timedelta(hours=3)
        rows=flatten_story(tiny_thread(),"BTCUSDT","bitcoin",start,end,100)
        self.assertEqual([r["node_id"] for r in rows],[100,101,102])

    def test_search_can_be_stubbed_without_the_network(self):
        def service(path,params=None):
            if path=="/search_by_date":
                return {"hits":[{"id":100,"created_at":event_time(0)}]}
            return tiny_thread()
        rows,logs=harvest(["BTCUSDT"],"2025-06","2025-06",1,20,0,service)
        self.assertEqual(len(rows),4)
        self.assertEqual(logs[0]["stories_fetched"],1)

try:
    import numpy as np
    import pandas as pd
    from temporal_graph_features import (load_events,make_hourly_graph,
                                         graph_windows,GRAPH_DIM)
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,"numpy/pandas not installed")
class GraphCausalityTests(unittest.TestCase):
    def get_fixture(self):
        stamp=pd.date_range("2025-06-15",periods=7,freq="h",tz="UTC")
        df=pd.DataFrame({"timestamp":stamp})
        raw=flatten_story(tiny_thread(),"BTCUSDT","bitcoin",
                          stamp[0].to_pydatetime(),
                          (stamp[-1]+pd.Timedelta("1h")).to_pydatetime(),100)
        events=pd.DataFrame(raw)
        events["created_at"]=pd.to_datetime(events.created_at,utc=True)
        return df,events

    def test_future_replies_do_not_change_past_graph(self):
        df,events=self.get_fixture()
        all_graph=make_hourly_graph(df,events)
        past=events[events.created_at<df.timestamp.iloc[3]].copy()
        past_graph=make_hourly_graph(df,past)
        np.testing.assert_array_equal(all_graph.hourly[:3],past_graph.hourly[:3])
        self.assertEqual(all_graph.valid_edge_count,3)
        self.assertEqual(past_graph.valid_edge_count,2)

    def test_directed_adj_and_no_future_at_decision(self):
        df,events=self.get_fixture()
        full=make_hourly_graph(df,events)
        axis=np.arange(len(df),dtype=float)
        graph,adj,idx=graph_windows(full,axis,[2],scales=(1,),length=3)
        self.assertEqual(graph.shape,(1,1,GRAPH_DIM,3))
        self.assertGreater(float(adj[0,0,1,0]),0)
        self.assertGreater(float(adj[0,0,2,1]),0)
        self.assertEqual(float(adj[0,0,0,2]),0)
        self.assertEqual(int(idx[0]),2)

    def test_missing_or_backwards_event_rejected(self):
        df,events=self.get_fixture()
        events.loc[1,"created_at"]=pd.Timestamp("2025-06-14T23:55:00Z")
        with self.assertRaises(ValueError):
            make_hourly_graph(df,events)

try:
    import torch
    from graph_fusion import PriceGraphFusion
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False

@unittest.skipUnless(HAS_TORCH,"torch unavailable")
class NeuralIntegrationTests(unittest.TestCase):
    def test_fusion_heads_and_graph_ablation(self):
        torch.manual_seed(42)
        net=PriceGraphFusion().eval()
        price=torch.randn(2,3,4,48)
        graph=torch.randn(2,3,GRAPH_DIM,48)
        adj=torch.eye(48)[None,None].repeat(2,3,1,1)
        adj[:,0,1,0]=.2
        adj[:,0,1,1]=.8
        with torch.no_grad():
            out=net(price,graph,adj,True)
            off=net(price,graph,adj,False)
        self.assertEqual(tuple(out[0].shape),(2,17))
        self.assertEqual(tuple(out[1].shape),(2,))
        self.assertEqual(tuple(out[2].shape),(2,))
        self.assertFalse(torch.allclose(out[0],off[0]))
        zero=net(price,torch.zeros_like(graph),adj,True)
        a=net(price,graph,adj,False)
        self.assertTrue(torch.allclose(zero[0],a[0]))

if __name__=="__main__":unittest.main()
