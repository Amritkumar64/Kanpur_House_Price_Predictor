import pickle
import pandas as pd
import gradio as gr

# Load the trained model
with open('RidgeModel.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_price(area, bedrooms, bathrooms, locality):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, locality]],
                              columns=['area', 'bedrooms', 'bathrooms', 'locality'])
    prediction = model.predict(input_data)
    return f"Estimated Price: ₹{prediction[0]:,.2f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Textbox(label="Locality (e.g. Civil Lines, Kalyanpur)")
    ],
    outputs="text",
    title="Kanpur House Price Predictor",
    description="Enter house details to estimate price in Kanpur"
)

demo.launch()