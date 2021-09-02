# agricultural-production-optimization
The model can achieve the optimum production plan of an agricultural region combining in one utility function different conflicting criteria .

Machine learning is used to analyze data 

These data set suggesting crops according to the given data set of soil and using the predicition of weather
 
In given dataset we have different value of Nitrogen,phosphorous,potassium,temperature,humidity,ph,rainfall

Machine learning process is used for prediction of best soil,weather and climatic condition and  get the best result for our crops grow 

Example 
Nitrogen=90,phosphorous=40,potassium=40,temperature=20,humidity=80,ph=7,rainfall=200

input=> predicition = model.predict((np.array([[90,40,40,20,80,7,200]])))
        print("the suggested crop for given climatic condition id :", predicition)'
        
output=>  the suggested crop for given climatic condition id : ['rice'] => predicition of best soil , weather and climatic condition

Result => Through all these process we are getting the best predicition to grow the crops in depending soil ,weather and climatic condition

   
