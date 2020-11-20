# Dependencies
from flask import Flask, request, jsonify, render_template
import requests
import joblib
import traceback
import pandas as pd
import numpy as np
from io import StringIO


# Your API definition
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            # the key name is parameter
            f = request.files['attachment']
            # save the file
            f.save('%s'%f.filename)
            # read the file
            df=pd.read_csv('%s'%f.filename)
            df.drop(['GameId'], axis = 1, inplace = True)
            
            query = df.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            
            df['predicted']= prediction
            df.sort_values(by='predicted',inplace=True,ascending=False)

            top_players= df.drop_duplicates("PlayerId")
            top_players.to_csv("normal.csv",index=False)
            playerId = list(top_players['PlayerId'].head())
            
            return render_template('index.html',members=playerId)
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
# Applying average method
@app.route('/predict1', methods=['POST'])
def predictavg():
    lr = joblib.load("model1.pkl") # Load "model.pkl"
    print ('Model1 loaded')
    model_columns = joblib.load("model_columns1.pkl") # Load "model_columns.pkl"
    print ('Model columns 1 loaded')
    if lr:
        try:
            # the key name is parameter
            f = request.files['attachment']
            # save the file
            f.save('%s'%f.filename)
            # read the file
            dataset=pd.read_csv('%s'%f.filename)
            
            player_id = dataset['PlayerId']
            dataset.drop(['GameId'], axis = 1, inplace = True)

            unique_player_data = []
            # create list of player id
            value = dataset['PlayerId']
            x = np.array(value)
            # compute average score of each attribute of individual player
            unique_player_id,counts = np.unique(x,return_counts=True)
            j=0
            for i in counts:
              k=j
              j= j+i
              val = dataset.loc[k:j].mean()
              unique_player_data.append(np.array(val))
              
            new_dataframe = pd.DataFrame(data=unique_player_data,columns=dataset.columns)
            new_dataframe['PlayerId'] = new_dataframe['PlayerId'].astype(int) 
            
            query = new_dataframe.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            
            new_dataframe['predicted']= prediction
            new_dataframe.sort_values(by='predicted',inplace=True,ascending=False)

            top_players= new_dataframe.drop_duplicates("PlayerId")
            top_players.to_csv("avgpred.csv",index=False)
            playerId = list(top_players['PlayerId'].head())
            
            return render_template('index.html',avgmembers=playerId)
        
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predictjson', methods=['POST'])
def predictjson():
    if lr:
        try:
            successfulDribbling = request.form.get('dribbling')
            successfulPass = request.form.get('pass')
            playerAttackingScore = request.form.get('attacking')
            playerDefendingScore = request.form.get('defending')
            playerTeamPlayScore = request.form.get('teamplay')

            x = [[successfulDribbling,successfulPass,playerAttackingScore,playerDefendingScore,playerTeamPlayScore]]
            query = pd.DataFrame(data=x,columns=model_columns)
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = lr.predict(query)
            print(type(prediction))
            return render_template('index.html',prediction=str(prediction[0]))

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 1234 # If you don't provide any port the port will be set to 12345
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
