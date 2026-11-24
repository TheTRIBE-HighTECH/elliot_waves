"""Unit tests: real reply edges, exact causal temporal joins and graph clock."""
import unittest, datetime as dt
import numpy as np
import pandas as pd
from hn_graph import extract_thread, range_epoch
from graph_fusion import graph_features_hourly,graph_clock,sample_graph

class ThreadTests(unittest.TestCase):
    def test_parent_reply_tree(self):
        root={"id":100,"created_at_i":1704067200,"children":[
            {"id":101,"parent_id":100,"created_at_i":1704067260,"children":[
                {"id":102,"parent_id":101,"created_at_i":1704067380,"children":[]}]}]}
        rows=extract_thread(root,"bitcoin")
        self.assertEqual([x["id"] for x in rows],[100,101,102])
        self.assertEqual([x["depth"] for x in rows],[0,1,2])
        self.assertEqual([x["parent_id"] for x in rows],["",100,101])
    def test_invalid_claimed_parent(self):
        root={"id":100,"created_at_i":1704067200,"children":[
            {"id":101,"parent_id":55,"created_at_i":1704067260,"children":[]}]}
        with self.assertRaises(ValueError):extract_thread(root,"bitcoin")
    def test_month_end(self):
        start,end=range_epoch(2024,2)
        self.assertEqual(end-start,29*86400)

class CausalityTests(unittest.TestCase):
    def setup_data(self):
        begin=pd.Timestamp("2024-01-01T00:00:00Z")
        times=pd.date_range(begin,periods=520,freq="h",tz="UTC")
        candles=pd.DataFrame({"timestamp":times})
        def event(id,parent,story,at,depth,kind):
            return {"id":id,"parent_id":parent,"story_id":story,
                    "time_utc":times[at].isoformat(),"depth":depth,
                    "kind":kind,"topic":"bitcoin"}
        past=[event(300,"",300,20,0,"story"),
              event(301,300,300,21,1,"comment"),
              event(302,301,300,22,2,"comment")]
        future=event(303,302,300,450,3,"comment")
        return candles,pd.DataFrame(past),pd.DataFrame(past+[future])
    def test_future_events_cannot_change_previous_graph_history(self):
        df,a,b=self.setup_data()
        g1=graph_features_hourly(df,a);g2=graph_features_hourly(df,b)
        self.assertTrue(np.array_equal(g1[:450],g2[:450]))
        self.assertGreater(g2[450,1],0)
    def test_graph_clock_always_increasing(self):
        df,a,b=self.setup_data()
        g=graph_features_hourly(df,b)
        tau,speed=graph_clock(g)
        self.assertTrue(np.all(np.diff(tau)>0))
        zero,vel=graph_clock(g,alpha=0)
        self.assertTrue(np.array_equal(zero,np.arange(len(g),dtype=float)))
    def test_graph_sampling_never_uses_future_events(self):
        df,a,b=self.setup_data()
        g1=graph_features_hourly(df,a);g2=graph_features_hourly(df,b)
        t=np.arange(len(g1),dtype=float)
        x=sample_graph(g1,t,[430]);y=sample_graph(g2,t,[430])
        self.assertEqual(x.shape,(1,3,10,48))
        self.assertTrue(np.array_equal(x,y))

if __name__=="__main__":unittest.main()
