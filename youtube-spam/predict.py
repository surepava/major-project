import joblib

usa = joblib.load('model1.pkl')

k = usa.predict(["+447935454150 lovely girl talk to me"])
print(k)