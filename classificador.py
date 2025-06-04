import numpy
import pandas as pd
import scipy
import spacy
import sklearn
import nltk
import os
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import word_tokenize
from Modules.lemmatizer import Lemmatizer
from Modules.stopword import Stopword_RMV
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

class Classificador:

    def __init__(self, nome_lexico = None):
        # Defines 3 dictionaries, one for the tags and the other 2 for the words:
        # - self.tags has each tag classification (-1 neg, 0 neu, 1 pos)
        # - self.classified_dictionary has each word classification (-1 neg, 0 neu, 1 pos)
        # - self.dictionary has each word and its tags

        # Open Lexicon
        
        if(nome_lexico == None):
            nome_lexico = os.path.join("Docs", "LIWC.txt")
        f = open(nome_lexico, "r")

        # Reading Tags

        if(f.readline() == "%\n"):

            self.tags = dict()

            line = f.readline()
            while(len(line) > 2):
                words = line.split()
                self.tags[int(words[0])] = int(words[2])
                # print(words)
                line = f.readline()
            # print(self.tags)
        else:
            print("error")

        # Reading Dictionary

        line = f.readline()

        self.classified_dictionary = dict()
        self.dictionary = dict()

        while(len(line) > 0):
            words = line.split()
            # print(words)
            sum = 0
            for i in words[1:]:
                sum = sum + self.tags[int(i)]
            if(sum > 3):
                sum = 3
            if(sum < -3):
                sum = -3 
            self.classified_dictionary[words[0]] = sum
            self.dictionary[words[0]] = [int(word) for word in words[1:]]
            line = f.readline()
        
        # print(self.dictionary)

    def classify_token(self, word):
        # Receives a token and returns -1 if negative, 0 if neutral and 1 if positive
        if(not(word in self.classified_dictionary)):
            return 0
        return self.classified_dictionary[word]
    
    def check_negation_token(self,word):
        # Check if this token indicates negation for the next one, returns 1 if false and -1 if true
        if(not(word in self.dictionary)):
            return 1

        # Negations
        if(19 in self.dictionary[word]):
            return -1
        
        return 1

    def check_intensity_token(self, word):
        # Check if this token indicates intensity for the next one, returns 1 if false and 2 if true
        if(not(word in self.dictionary)):
            return 1
        
        # Quantifiers
        if(20 in self.dictionary[word]):
            return 2

        return 1
        

    # Function: returns tokens of sentence
    def tokenization(self, sentence):

        self.tokens = word_tokenize(str(sentence).lower(), language='portuguese') # tokenizer

    def remove_stopword(self):
        # Remove tokens that dont help in classification (words with specific tags can be taken out)
        rmv_sw = Stopword_RMV()

        self.tokens = rmv_sw.remove_stopwords(self.tokens)

    # Receives a text and returns the same text after lemmatization is applyed, not used in classical approaches
    def lemmatization(self, text):
        lem = Lemmatizer()
        return lem.lemmatize_text(text)

    def classic_classification(self, text = "Nothing here\n", text_file = None, x_print = False):
        # Receives a text or a file name and classify that text in Positive, Neutral or Negative

        if(text_file != None):
            # Read file into text
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()

        # Tokenization
        self.tokenization(text)

        if(x_print):
            print("Sentence Tokenized:")
            print(self.tokens)

        # Remove Stop Word
        self.remove_stopword()

        if(x_print):
            print("Stopwords Removed:")
            print(self.tokens)
            print("Classification Values:")
            print([self.classify_token(i) for i in self.tokens])

        # Classify tokens
        intensity = 1
        value = 0
        for i in self.tokens:
            value += intensity * self.classify_token(i)
            intensity = self.check_intensity_token(i)
            intensity *= self.check_negation_token(i)


        # Print Results
        if(value > 0):
            return "Positivo"
        
        if(value < 0):
            return "Negativo"

        if(value >= 0 and value <= 0):
            return "Neutro"
        
    def pretrained_model_classification(self, texto):

        # Carregar modelo e tokenizer
        model_name = "lipaoMai/BERT-sentiment-analysis-portuguese-with-undersampling-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Tokenizar
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)

        # Fazer previsão
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probs)

        # Mostrar resultado
        sentimentos = ['Negativo', 'Neutro', 'Positivo']
        #print(f"Sentimento previsto: {sentimentos[predicted]}")

        return sentimentos[predicted]

    def model_training_classification(self, nome_arquivo, num_each_class=1000):
        # Carregar o dataset
        df = pd.read_csv(nome_arquivo)  # deve conter colunas 'Text' e 'Classificacao'

        # Balancear o número de exemplos por classe
        df = df.groupby('Classificacao').apply(lambda x: x.sample(n=num_each_class, random_state=42)).reset_index(drop=True)

        # Separar em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Classificacao"], test_size=0.2, random_state=42)

        # Definir o pipeline e os parâmetros para grid search
        pipelines = {
            'LogisticRegression': Pipeline([
                ('vect', TfidfVectorizer()), 
                ('clf', LogisticRegression(max_iter=1000))
            ]),
            'RandomForest': Pipeline([
                ('vect', CountVectorizer()), 
                ('clf', RandomForestClassifier())
            ]),
            'LinearSVC': Pipeline([
                ('vect', TfidfVectorizer()), 
                ('clf', LinearSVC())
            ])
        }

        param_grids = {
            'LogisticRegression': {
                'vect__ngram_range': [(1, 1), (1, 2)],
                'vect__max_features': [3000, 5000],
                'clf__C': [0.1, 1, 10]
            },
            'RandomForest': {
                'vect__ngram_range': [(1, 1)],
                'vect__max_features': [3000],
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 20]
            },
            'LinearSVC': {
                'vect__ngram_range': [(1, 1), (1, 2)],
                'vect__max_features': [3000, 5000],
                'clf__C': [0.1, 1, 10]
            }
        }

        best_models = {}

        for name in pipelines:
            print(f"\nRodando GridSearch para {name}...")
            grid = GridSearchCV(pipelines[name], param_grids[name], cv=3, n_jobs=-1, scoring='f1_macro')
            grid.fit(X_train, y_train)
            
            print(f"\nMelhores parâmetros para {name}: {grid.best_params_}")
            y_pred = grid.predict(X_test)
            print(f"\nRelatório de Classificação para {name}:\n", classification_report(y_test, y_pred))

            best_models[name] = grid.best_estimator_

        # Salvar o melhor modelo (exemplo: LogisticRegression)
        final_model = best_models['LogisticRegression']
        joblib.dump(final_model, 'final_sentiment_model.pkl')

        print("\nModelo salvo como 'final_sentiment_model.pkl'.")

    def model_classification(self, texto, pipeline):
        return pipeline.predict([texto])[0]
