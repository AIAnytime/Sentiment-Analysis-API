from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from textblob import TextBlob
import uvicorn

app = FastAPI(title="Sentiment Analysis API")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.get("/sentiment-analysis/{text}")
def sentiment_analysis(text: str):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity
    }

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000)
