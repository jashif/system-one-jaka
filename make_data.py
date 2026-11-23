"""Generate a balanced synthetic routing benchmark; not real-world training data."""
import json, random
from pathlib import Path

CATEGORIES = {
 'Billing': ['I was charged twice for my order','My invoice amount is incorrect','I need a refund for a duplicate payment','The payment failed but money was taken','My subscription renewal fee is wrong','I need a receipt for the transaction','Why was my credit card billed again','Please reverse an incorrect charge','The tax on my invoice looks wrong','I have a question about a refund','My payment reminder is incorrect','I was overcharged for the service'],
 'Technical': ['The application crashes when I log in','The website displays an internal server error','The mobile app freezes on startup','The password reset link returns a 500 error','The dashboard is not loading','The API responds with a timeout','The upload button does not work','The screen turns blank after sign in','The app has a software bug','I cannot connect to the server','The login page keeps crashing','The application shows a stack trace'],
 'Sales': ['I want a quote for a new enterprise plan','Can I get a product demo','What are your pricing packages for new customers','I want to purchase more licenses','Tell me about your premium features','Can I speak to someone about a new contract','I am interested in buying your product','What discounts are available for a bulk purchase','I would like to upgrade to the business tier','Can someone explain the product offering','I need a sales proposal','I want to compare your plans before buying'],
 'Account': ['Please change my profile display name','I want to update my email address','Please close my account','I need to change my contact details','How can I delete my user profile','I want to change my notification preferences','Please update my mailing address','I need to manage my account settings','My username needs to be changed','I want to deactivate my account','Please update the phone number on my profile','I need to change the account owner']
}
QUESTIONS=['Which team should handle this request?','Who should receive this customer message?','What is the best routing category?','Which department is responsible?']
PREFIXES=['','Customer says: ','Support ticket: ','Incoming request: ']
SUFFIXES=['',' Please help.',' Can you assist?',' I need support.']
OPTIONS=['Billing','Technical','Sales','Account']

def generate():
    rng=random.Random(2026)
    splits={'train':[],'val':[],'test':[]}
    for category,templates in CATEGORIES.items():
        for idx,base in enumerate(templates):
            split='train' if idx<8 else ('val' if idx<10 else 'test')
            count=12 if split=='train' else 4
            for _ in range(count):
                options=OPTIONS[:];rng.shuffle(options)
                context=rng.choice(PREFIXES)+base+rng.choice(SUFFIXES)
                splits[split].append({'context':context,'question':rng.choice(QUESTIONS),'options':options,'label':options.index(category)})
    root=Path(__file__).parent/'data';root.mkdir(exist_ok=True)
    for split,rows in splits.items():
        rng.shuffle(rows)
        with (root/f'{split}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        print(split,len(rows))
if __name__=='__main__':generate()
