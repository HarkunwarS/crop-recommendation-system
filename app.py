from flask import Flask, render_template, url_for,request
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import random


app=Flask(__name__)

@app.route("/")

def index():
    return render_template("crop.html", title="Crop Recommendation System")

@app.route("/cropstyle.css")
def css():
    return url_for('static', filename='cropstyle.css')

@app.route("/farm.jpg")
def img():
    return url_for('static', filename='farm.css')

@app.route("/recommendation", methods=['POST'])
def recommendation():
    rain=request.form['rain']
    temp=request.form['temp']
    hum=request.form['humid']
    
    rainfall = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rainfall')
    temperature = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'humidity')
    yield_output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'yield_output')

    rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0.25, 0.5])
    rainfall['medium'] = fuzz.trimf(rainfall.universe, [0.25, 0.5, 0.75])
    rainfall['high'] = fuzz.trimf(rainfall.universe, [0.5, 0.75, 1])

    temperature['low'] = fuzz.trimf(temperature.universe, [0, 0.25, 0.5])
    temperature['medium'] = fuzz.trimf(temperature.universe, [0.25, 0.5, 0.75])
    temperature['high'] = fuzz.trimf(temperature.universe, [0.5, 0.75, 1])

    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0.25, 0.5])
    humidity['medium'] = fuzz.trimf(humidity.universe, [0.25, 0.5, 0.75])
    humidity['high'] = fuzz.trimf(humidity.universe, [0.5, 0.75, 1])

    yield_output['low'] = fuzz.trimf(yield_output.universe, [0, 0.25, 0.5])
    yield_output['medium'] = fuzz.trimf(yield_output.universe, [0.25, 0.5, 0.75])
    yield_output['high'] = fuzz.trimf(yield_output.universe, [0.5, 0.75, 1])

    #Rule base for Bajra
    Brule1 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['low'], yield_output['low'])
    Brule2 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Brule3 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['low'], yield_output['low'])
    Brule4 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Brule5 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Brule6 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['medium'], yield_output['low'])
    Brule7 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['high'], yield_output['low'])
    Brule8 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Brule9 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['high'], yield_output['medium'])
    Brule10 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['low'], yield_output['medium'])
    Brule11 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Brule12 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Brule13 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Brule14 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['medium'], yield_output['high'])
    Brule15 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Brule16 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Brule17 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['high'], yield_output['high'])
    Brule18 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['high'], yield_output['medium'])
    Brule19 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['low'], yield_output['medium'])
    Brule20 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Brule21 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['low'], yield_output['low'])
    Brule22 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Brule23 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Brule24 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Brule25 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Brule26 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['high'], yield_output['medium'])
    Brule27 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['high'], yield_output['low'])

    #Rule base for JOWAR
    Jrule1 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['low'], yield_output['low'])
    Jrule2 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Jrule3 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['low'], yield_output['low'])
    Jrule4 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Jrule5 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Jrule6 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['medium'], yield_output['low'])
    Jrule7 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['high'], yield_output['low'])
    Jrule8 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Jrule9 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['high'], yield_output['medium'])
    Jrule10 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['low'], yield_output['medium'])
    Jrule11 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Jrule12 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Jrule13 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Jrule14 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['medium'], yield_output['high'])
    Jrule15 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Jrule16 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Jrule17 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['high'], yield_output['high'])
    Jrule18 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['high'], yield_output['medium'])
    Jrule19 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['low'], yield_output['low'])
    Jrule20 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Jrule21 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['low'], yield_output['low'])
    Jrule22 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Jrule23 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Jrule24 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['medium'], yield_output['low'])
    Jrule25 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Jrule26 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Jrule27 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['high'], yield_output['low'])

    #Rule base for MAIZE
    Mrule1 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['low'], yield_output['low'])
    Mrule2 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Mrule3 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Mrule4 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Mrule5 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['medium'], yield_output['medium'])
    Mrule6 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Mrule7 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Mrule8 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['high'], yield_output['medium'])
    Mrule9 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['high'], yield_output['medium'])
    Mrule10 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['low'], yield_output['medium'])
    Mrule11 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Mrule12 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Mrule13 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Mrule14 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['medium'], yield_output['high'])
    Mrule15 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Mrule16 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['high'], yield_output['high'])
    Mrule17 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['high'], yield_output['high'])
    Mrule18 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['high'], yield_output['high'])
    Mrule19 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['low'], yield_output['low'])
    Mrule20 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Mrule21 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['low'], yield_output['low'])
    Mrule22 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Mrule23 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Mrule24 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Mrule25 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['high'], yield_output['low'])
    Mrule26 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Mrule27 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['high'], yield_output['low'])

    #Rule base for wheat
    Wrule1 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['low'], yield_output['low'])
    Wrule2 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['low'], yield_output['medium'])
    Wrule3 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Wrule4 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Wrule5 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['medium'], yield_output['medium'])
    Wrule6 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Wrule7 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Wrule8 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['high'], yield_output['medium'])
    Wrule9 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['high'], yield_output['low'])
    Wrule10 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['low'], yield_output['low'])
    Wrule11 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Wrule12 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['low'], yield_output['low'])
    Wrule13 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Wrule14 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['medium'], yield_output['high'])
    Wrule15 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['medium'], yield_output['high'])
    Wrule16 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Wrule17 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['high'], yield_output['medium'])
    Wrule18 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['high'], yield_output['high'])
    Wrule19 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['low'], yield_output['high'])
    Wrule20 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['low'], yield_output['high'])
    Wrule21 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['low'], yield_output['high'])
    Wrule22 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['medium'], yield_output['high'])
    Wrule23 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['medium'], yield_output['medium'])
    Wrule24 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['medium'], yield_output['high'])
    Wrule25 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['high'], yield_output['high'])
    Wrule26 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['high'], yield_output['high'])
    Wrule27 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['high'], yield_output['high'])

    #Rule base for RICE
    Rrule1 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['low'], yield_output['low'])
    Rrule2 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Rrule3 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Rrule4 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['medium'], yield_output['medium'])
    Rrule5 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Rrule6 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Rrule7 = ctrl.Rule(rainfall['low'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Rrule8 = ctrl.Rule(rainfall['low'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Rrule9 = ctrl.Rule(rainfall['low'] & temperature['high'] & humidity['high'], yield_output['low'])
    Rrule10 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['low'], yield_output['low'])
    Rrule11 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Rrule12 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['low'], yield_output['low'])
    Rrule13 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Rrule14 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['medium'], yield_output['high'])
    Rrule15 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['medium'], yield_output['high'])
    Rrule16 = ctrl.Rule(rainfall['medium'] & temperature['low'] & humidity['high'], yield_output['medium'])
    Rrule17 = ctrl.Rule(rainfall['medium'] & temperature['medium'] & humidity['high'], yield_output['medium'])
    Rrule18 = ctrl.Rule(rainfall['medium'] & temperature['high'] & humidity['high'], yield_output['high'])
    Rrule19 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['low'], yield_output['low'])
    Rrule20 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['low'], yield_output['low'])
    Rrule21 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['low'], yield_output['medium'])
    Rrule22 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['medium'], yield_output['low'])
    Rrule23 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['medium'], yield_output['low'])
    Rrule24 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['medium'], yield_output['medium'])
    Rrule25 = ctrl.Rule(rainfall['high'] & temperature['low'] & humidity['high'], yield_output['low'])
    Rrule26 = ctrl.Rule(rainfall['high'] & temperature['medium'] & humidity['high'], yield_output['low'])
    Rrule27 = ctrl.Rule(rainfall['high'] & temperature['high'] & humidity['high'], yield_output['low'])

    # Create the control system
    Bcrop_prediction_ctrl = ctrl.ControlSystem([Brule1, Brule2, Brule3,Brule4,Brule5,Brule6,Brule7,Brule8,Brule9,Brule10,Brule11,Brule12,Brule13,Brule14,Brule15,Brule16,Brule17,Brule18,Brule19,Brule20,Brule21,Brule22,Brule23,Brule24,Brule25,Brule26,Brule27])
    Jcrop_prediction_ctrl = ctrl.ControlSystem([Jrule1, Jrule2, Jrule3,Jrule4,Jrule5,Jrule6,Jrule7,Jrule8,Jrule9,Jrule10,Jrule11,Jrule12,Jrule13,Jrule14,Jrule15,Jrule16,Jrule17,Jrule18,Jrule19,Jrule20,Jrule21,Jrule22,Jrule23,Jrule24,Jrule25,Jrule26,Jrule27])
    Mcrop_prediction_ctrl = ctrl.ControlSystem([Mrule1, Mrule2, Mrule3,Mrule4,Mrule5,Mrule6,Mrule7,Mrule8,Mrule9,Mrule10,Mrule11,Mrule12,Mrule13,Mrule14,Mrule15,Mrule16,Mrule17,Mrule18,Mrule19,Mrule20,Mrule21,Mrule22,Mrule23,Mrule24,Mrule25,Mrule26,Mrule27])
    Rcrop_prediction_ctrl = ctrl.ControlSystem([Rrule1, Rrule2, Rrule3,Rrule4,Rrule5,Rrule6,Rrule7,Rrule8,Rrule9,Rrule10,Rrule11,Rrule12,Rrule13,Rrule14,Rrule15,Rrule16,Rrule17,Rrule18,Rrule19,Rrule20,Rrule21,Rrule22,Rrule23,Rrule24,Rrule25,Rrule26,Rrule27])
    Wcrop_prediction_ctrl = ctrl.ControlSystem([Wrule1, Wrule2, Wrule3,Wrule4,Wrule5,Wrule6,Wrule7,Wrule8,Wrule9,Wrule10,Wrule11,Wrule12,Wrule13,Wrule14,Wrule15,Wrule16,Wrule17,Wrule18,Wrule19,Wrule20,Wrule21,Wrule22,Wrule23,Wrule24,Wrule25,Wrule26,Wrule27])

    # Create a control system simulation
    Bcrop_prediction = ctrl.ControlSystemSimulation(Bcrop_prediction_ctrl)
    Jcrop_prediction = ctrl.ControlSystemSimulation(Jcrop_prediction_ctrl)
    Mcrop_prediction = ctrl.ControlSystemSimulation(Mcrop_prediction_ctrl)
    Rcrop_prediction = ctrl.ControlSystemSimulation(Rcrop_prediction_ctrl)
    Wcrop_prediction = ctrl.ControlSystemSimulation(Wcrop_prediction_ctrl)

        #Taking the values of each input variable
    if(rain=='low'):
        rainfall=random.uniform(0,0.3)
    elif(rain=='medium'):
        rainfall=random.uniform(0.4,0.6)
    else:
        rainfall=random.uniform(0.7,1)
        
    if(temp=='low'):
        temperature=random.uniform(0,0.3)
    elif(temp=='medium'):
        temperature=random.uniform(0.4,0.6)
    else:
        temperature=random.uniform(0.7,1)
        
    if(hum=='low'):
        humidity=random.uniform(0,0.3)
    elif(hum=='medium'):
        humidity=random.uniform(0.4,0.6)
    else:
        humidity=random.uniform(0.7,1)
        
    print('Rainfall: ',rainfall)
    print('Humidity: ',humidity)
    print('Temperature: ',temperature)

    #Bajra
    Bcrop_prediction.input['rainfall'] = rainfall
    Bcrop_prediction.input['temperature'] = temperature
    Bcrop_prediction.input['humidity'] = humidity

    #Jowar
    Jcrop_prediction.input['rainfall'] = rainfall
    Jcrop_prediction.input['temperature'] = temperature
    Jcrop_prediction.input['humidity'] = humidity

    #Maize
    Mcrop_prediction.input['rainfall'] = rainfall
    Mcrop_prediction.input['temperature'] = temperature
    Mcrop_prediction.input['humidity'] = humidity

    #Rice
    Rcrop_prediction.input['rainfall'] = rainfall
    Rcrop_prediction.input['temperature'] = temperature
    Rcrop_prediction.input['humidity'] = humidity

    #Wheat
    Wcrop_prediction.input['rainfall'] = rainfall
    Wcrop_prediction.input['temperature'] = temperature
    Wcrop_prediction.input['humidity'] = humidity
    

    # Compute the output for each crop
    Bcrop_prediction.compute()
    Jcrop_prediction.compute()
    Mcrop_prediction.compute()
    Rcrop_prediction.compute()
    Wcrop_prediction.compute()

    # Print the output
    print("Bajra: ",Bcrop_prediction.output['yield_output'])
    print("Jowar: ",Jcrop_prediction.output['yield_output'])
    print("Maize: ",Mcrop_prediction.output['yield_output'])
    print("Rice: ",Rcrop_prediction.output['yield_output'])
    print("Wheat: ",Wcrop_prediction.output['yield_output'])

    #Comparing the yield of all the crops and selecting the best one

    max_yield=0.0

    yields=[Bcrop_prediction.output['yield_output'],Jcrop_prediction.output['yield_output'],Mcrop_prediction.output['yield_output'],Rcrop_prediction.output['yield_output'],Wcrop_prediction.output['yield_output']]

    for i in yields:
        if(i>max_yield):
            max_yield=i

    if(max_yield==Bcrop_prediction.output['yield_output']):
        crop='Bajra'
    elif(max_yield==Jcrop_prediction.output['yield_output']):
        crop='Jowar'
    elif(max_yield==Mcrop_prediction.output['yield_output']):
        crop='Maize'
    elif(max_yield==Rcrop_prediction.output['yield_output']):
        crop='Rice'
    elif(max_yield==Wcrop_prediction.output['yield_output']):
        crop='Wheat'
        
    print('Maximum Yield: ',max_yield)
        
    return render_template('recommendation.html', crop=crop)


if __name__=='__main__':
    app.run()