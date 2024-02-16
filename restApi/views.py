from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import pandas as pd

@api_view(['POST'])
def predict(request):
    model = joblib.load('./decision_tree_model.pkl')
    input_data_str = request.data.get('input_data')
    print(input_data_str)
    if input_data_str is None:
        return Response({'error': 'Input data not provided'}, status=400)
    
    input_data = [float(x) for x in input_data_str]
    
    input_array = [input_data]
    
    prediction = model.predict(input_array)
    
    return Response({'prediction': prediction.tolist()})
