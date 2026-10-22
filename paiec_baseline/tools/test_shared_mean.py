import unittest
from shared_mean.model import predict,from_counts


class SharedMeanTests(unittest.TestCase):
    def test_scope_and_visible_config_identity(self):
        target=[{"normalized_name":"a"},{"benchmark_id":"x"}]
        own=[target,1]
        peer=[[{"normalized_name":"b"},{"benchmark_id":"x"}],0]
        unrelated=[[{"normalized_name":"b"},{"benchmark_id":"y"}],0]
        self.assertEqual(predict(target,[]),.5)
        self.assertAlmostEqual(predict(target,[own]),2/3)
        self.assertEqual(predict(target,[own,unrelated]),predict(target,[own]))
        self.assertAlmostEqual(predict(target,[own,peer]),from_counts(1,1,0,1))
        self.assertLess(predict(target,[own,peer]),predict(target,[own]))
        duplicate=[[dict(target[0]),{"benchmark_id":"x","item_content":"another item"}],0]
        self.assertEqual(predict(target,[own,duplicate]),.5)
        self.assertEqual(predict(target,[]),.5)


if __name__=="__main__":unittest.main()
