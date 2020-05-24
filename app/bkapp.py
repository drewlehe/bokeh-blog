import bokeh
from bokeh.models import Button, Paragraph, TextInput, WidgetBox
from bokeh.plotting import curdoc, figure
from bokeh.embed import server_document
from bokeh.server.server import Server
from clean import clean

import pandas as pd
import joblib

vocab = joblib.load('feature')
model = joblib.load('xgbooster')



from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer(min_df=2, max_df=0.6, smooth_idf=True,
                              norm = 'l2', ngram_range=[1,2], max_features=125000,
                              decode_error="replace", vocabulary=vocab)
transformer.fit_transform(vocab)

textin = TextInput(title = "Submit Blog Post:")
button = Button(label="Submit", button_type="success")
p = Paragraph(text="Blog entry here")
def update_data(event):
	data = str(textin.value)
	vector = transformer.transform([' '.join(clean(data))])
	result = model.predict(vector)
	if int(result) == 1:
	    pred_text = 'Male'
	else:
	    pred_text = 'Female'
	output = {'prediction': pred_text}
	p.text = "{}".format(output)
button.on_click(update_data)
box = WidgetBox(children = [textin, button, p])
curdoc().add_root(box)
