import requests

class Predict():
    def __init__(self)-> None:
        self.url="http://127.0.0.1:8000/"
        self.data={"val1":0.02729,	"val2":0.0,	"val3":8.07,	"val4":0.0,
          "val5":0.469,	"val6":7.185,	"val7":61.1,	"val8":4.9671,	"val9":2.0,	"val10":242.0,	"val11":17.8,
          "val12":392.83,	"val13":4.03}

    def predict(self):
        

        response=requests.post(f"{self.url}/predict", json=self.data)

        data = response.json()

        # If your API returns a float prediction
        print("\n","="*34,data,"="*34,"\n",sep="\n")



if __name__=="__main__":
    predict_obj=Predict()
    predict_obj.predict()