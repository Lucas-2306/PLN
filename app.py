from flask import Flask, request, jsonify, render_template
from classificador import Classificador
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import praw
from datetime import datetime, timedelta
import joblib
from langdetect import detect

app = Flask(__name__)

# --- Reddit API Setup (PRAW) ---
def fetch_reddit_comments(subreddit = 'all', palavra='', periodo=''):
    reddit = praw.Reddit(client_id='Gw-kGHFca7xMD6ibdMHuIg',
                         client_secret='o_5CQ48lubVpOodH2hTCeGr-pFSI0g',
                         user_agent="my_reddit_scraper/0.1 by my_username")

    if periodo == 'last_week':
        time_filter = 'week'
    elif periodo == 'last_month':
        time_filter = 'month'
    elif periodo == 'last_year':
        time_filter = 'year'
    else:
        time_filter = 'all'

    # Search across all subreddits
    search_results = reddit.subreddit(subreddit).search(palavra, sort='new', time_filter=time_filter, limit=20)

    comments = []
    for submission in search_results:
        if detect(submission.title) == 'pt':
            comments.append({
                'title': submission.title,
                'url': submission.url,
                'score': submission.score,
                'created': datetime.utcfromtimestamp(submission.created_utc)
            })
    
    return comments

def fetch_gdelt_news(palavra='', periodo=''):
    from datetime import datetime, timedelta
    import requests

    query = palavra if palavra else '*'
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={query}&mode=ArtList&format=json&maxrecords=5&sort=DateDesc"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 429:
            print("⚠️ Erro 429: limite da GDELT atingido")
            return []
        elif response.status_code != 200:
            print(f"Erro GDELT: {response.status_code}")
            return []

        data = response.json()
    except Exception as e:
        print(f"Erro na chamada GDELT: {e}")
        return []

    articles = data.get('articles', [])
    resultados = []

    for article in articles:
        pub_str = article.get('seendate', '')
        try:
            pub_date = datetime.strptime(pub_str, "%Y%m%d%H%M%S")
        except:
            continue

        if periodo == 'last_week' and pub_date < datetime.utcnow() - timedelta(weeks=1):
            continue
        if periodo == 'last_month' and pub_date < datetime.utcnow() - timedelta(weeks=4):
            continue

        resultados.append({
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'published': pub_date
        })

    return resultados

# --- Disqus API Setup ---
def fetch_disqus_comments(palavra='', periodo=''):
    # Disqus API key and API base URL
    api_key = 'your_disqus_api_key'  # Replace with your Disqus API key
    forum = 'your_forum'  # Replace with your Disqus forum name
    url = f'https://disqus.com/api/3.0/forums/listThreads.json?api_key={api_key}&forum={forum}'

    # Request threads from Disqus
    response = requests.get(url)
    data = response.json()

    comments = []
    if data['code'] == 0:
        for thread in data['response']:
            if palavra.lower() in thread['title'].lower():
                # Filter by 'periodo' (time period)
                # Assuming threads have a 'created_at' field for the publish date
                created_time = datetime.strptime(thread['created_at'], "%Y-%m-%d %H:%M:%S")
                if periodo:
                    if periodo == 'last_week' and created_time >= datetime.utcnow() - timedelta(weeks=1):
                        comments.append({
                            'title': thread['title'],
                            'url': thread['link'],
                            'author': thread['author']['name'],
                            'created': created_time
                        })
                    elif periodo == 'last_month' and created_time >= datetime.utcnow() - timedelta(weeks=4):
                        comments.append({
                            'title': thread['title'],
                            'url': thread['link'],
                            'author': thread['author']['name'],
                            'created': created_time
                        })
                else:
                    comments.append({
                        'title': thread['title'],
                        'url': thread['link'],
                        'author': thread['author']['name'],
                        'created': created_time
                    })
    
    return comments

# --- Function to Generate Pie Chart ---
def generate_pie_chart(classificacoes):
    counts = [len(classificacoes['Positivo']), len(classificacoes['Negativo']), len(classificacoes['Neutro'])]

    # Check if all counts are zero
    if sum(counts) == 0:
        # Return an empty or placeholder image, or handle gracefully
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.axis('off')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    fig, ax = plt.subplots()
    ax.pie(counts,
           labels=['Positivo', 'Negativo', 'Neutro'],
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    plt.close(fig)  # Close figure to free memory

    return img_base64

# --- Function to Scrape, Filter, and Classify Data ---
def scrape_and_classify_data(palavra='', periodo=''):
    print("Buscando Reddit...")
    reddit_comments = fetch_reddit_comments('all', palavra, periodo)
    
    print("Buscando GDELT...")
    gdelt_news = fetch_gdelt_news(palavra, periodo)

    print("Combinando dados...")
    all_data = reddit_comments + gdelt_news

    print(f"Total de textos a classificar: {len(all_data)}")
    classificacoes = {'Positivo': [], 'Negativo': [], 'Neutro': []}
    classificador = Classificador("Docs/LIWC.txt")
    model_pipeline = joblib.load('final_sentiment_model.pkl')

    for i, item in enumerate(all_data):
        texto = item.get('title', '')
        print(f"Classificando {i+1}/{len(all_data)}: {texto[:40]}...")
        classificacao = classificador.model_classification(texto, model_pipeline)
        classificacoes[classificacao].append(texto)

    img_base64 = generate_pie_chart(classificacoes)
    return classificacoes, img_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classificar', methods=['POST'])
def classificar():
    data = request.json
    texto = data.get('texto', '')

    classificador = Classificador("Docs/LIWC.txt")

    # Carregar pipeline salvo (modelo + vetorizador)
    model_pipeline = joblib.load('final_sentiment_model.pkl')

    classificacao = classificador.model_classification(texto, model_pipeline)

    return jsonify({'classificacao': classificacao})

@app.route('/scraping_classificar', methods=['POST'])
def scraping_classificar():
    # Get the 'palavra' and 'periodo' from the request body
    data = request.json
    palavra = data.get('palavra', '')
    periodo = data.get('periodo', '')  # Default to empty string if not provided

    # Call the web_scraping_classificar function with both parameters
    classificacoes, img_base64 = scrape_and_classify_data(palavra, periodo)

    # Return the classifications and base64-encoded graph
    return jsonify({'classificacoes': classificacoes, 'grafico': img_base64})

if __name__ == '__main__':
    app.run(debug=True)