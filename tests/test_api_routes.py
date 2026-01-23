import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
#from src.routes.pydantic_model import ApiData
import requests

class TestRoutes(unittest.TestCase):
    def setUp(self)-> None:
        self.url="http://localhost:8000"
        self.data={"val1":0.02729,	"val2":0.0,	"val3":8.07,	"val4":0.0,
          "val5":0.469,	"val6":7.185,	"val7":61.1,	"val8":4.9671,	"val9":2.0,	"val10":242.0,	"val11":17.8,
          "val12":392.83,	"val13":4.03}

    def test_health(self):

        response=requests.get(f"{self.url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)

    def test_predict(self):
        

        response=requests.post(f"{self.url}/predict", json=self.data)
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # If your API returns a float prediction
        print("\n\n",data,"\n\n")
        self.assertIsInstance(data, dict)



if __name__=="__main__":
    unittest.main()