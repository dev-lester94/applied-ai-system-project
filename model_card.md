# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Les Find Music Finder 2.0

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

It recommends songs from a database off a user's query semantically. This is not design for real users. It is more for classroom exploration.

---

## 3. How the Model Works  

Explain your scoring approach in simple language. 

Each song has a description
A user queries for a song
The system compares the song and query semantically using a mathematically approach and computes a similarity score
Top 5 songs with scores closest to 1 are recommended

---

## 4. Data  

Describe the dataset the model uses.  

There are about 20 songs in the catalog.
Each song has a description
I did not remove any of the the original songs from my version of the app.
I would say their are songs in the classical genre that are missing in the data.

---

## 5. Strengths  

Where does your system seem to work well  

The system works well when the query semantically matches with the descriptions of the songs and when the query is indeed about music. The system is able to compute and recommend top 5 songs, no issues. 

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

The system does its best to recommended songs where the query isn't about music i.e. Can you help with filing taxes?. The system does compute a score where it is dissimilar and explains why the song was still recommended. I give a thumbs up for that. But logically, we should check query if it is about music and if it does not we can output I don't know.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

Tested a query about music: I want to listen to a electric pop song that is intense and danceable too. I saw the the cosine similarity for the top song was 0.80. I was surprised. I thought it would be higher. This lead me to test the second scenario.

The second scenario I tested was to make the query as the same as the song's description. The top song recommended had a cosine similarity of 0.86. I was very surprised again. I thought it would be exactly one.  After some evaluation, their were some differences in the way that we embed the description and the query. 

The third scenario I tested was to make a query Can you help with filing taxes?. Again it does try to recommend songs and justify an explanation on why a song was recommended this way. But logically, we should again check query if it is about music and if it does not we can output I don't know.


---

## 8. Future Work  

Ideas for how you would improve the model next.   

Ideas for Improvement
- Using a combination of content based criteria with semantics to recommend songs 
- Learn from user feedback to improve future recommendations.
- Use a larger and more diverse music dataset.
---

## 9. Personal Reflection  

A few sentences about your experience.  

  This was an awesome classroom project.
  I was able to design a different type of recommender.
  It was cool using the Gemini API for generation and embedding. 
  It allowed me to become a better engineer where I was more of a senior rather than a junior one.
  After working on this, I feel if making a real one for the users then we need to do a bit more research and planning to how to score the songs.
