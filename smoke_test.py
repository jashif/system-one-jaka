from predict import Predictor
p=Predictor()
examples=[('The application crashes when the user logs in','Technical'),('Customer was charged twice','Billing'),('I want to buy additional licenses','Sales'),('Please change my profile name','Account')]
correct=0
for context,expected in examples:
    result=p.predict(context,'Which team should handle this?',['Billing','Sales','Technical','Account'])
    print('expected:',expected,'predicted:',result['choice'],'probabilities:',result['probabilities'])
    correct+=result['choice']==expected
print(f'Smoke test: {correct}/{len(examples)} correct')
