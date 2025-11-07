# Projeto da Disciplina de PLN: Modelo de Classificação de Sentimentos de Comentários do Reddit

### Objetivo

Criar uma interface que permitisse o usuário definir um objeto (pessoa, assunto, produto) e um escopo para realização de um webscrapping de comentários usando a API do reddit.

Esses comentários relacionados aquele objeto são então classificados em positivo, negativo ou neutro, baseado em um modelo treinado no corpus B2W. O resultado final do sentimento da "população" acerca daquele objeto é então informado ao usuário.

Nossa ideia era buscar fazer um sistema semelhante ao do Brand24, porém bem mais simples. Permitindo ao usuário obter informações de como está o sentimento público acerca de algo.

### Como começar

**1.** Clonar o repositório
```bash
git clone https://github.com/LucasOF23/PLN.git
```

**2.** Instalar as dependências via pip install
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

## 📥 Download do modelo spaCy

Para que a lematização funcione corretamente, é necessário baixar o modelo de português do spaCy. Após instalar as dependências, execute o seguinte comando no terminal:

```bash
python -m spacy download pt_core_news_sm
```
