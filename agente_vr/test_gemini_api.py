import google.generativeai as genai

API_KEY = "AIzaSyDu2JisXeYwLAXqjHMzOHZlwZwy4U-DWnI"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Diga 'olá mundo' em português.")
    print("Gemini respondeu:", response.text)
except Exception as e:
    print("Erro Gemini:", e)
