## Model Applications

### Learning to Rank

**Tools**

- TensorFlow Ranking

**Starter Code**

PyTorch
[LambdaNet][1], [paper][2]

Tensorflow
[TF Ranking example with TFRecord, with self-attentive document interaction network][3]
[TF Portfolio Ranking][4]

Keras
[TF Keras][5]


**Approaches**

- Pair-wise Comparison
- List-wise Comparison

**Metrics**

Normalized Discounted Cumulative Gains (NDCG)

**Interaction Among Documents**

[Self-Attentive Document Interaction Networks for Permutation Equivariant Ranking][6]

**Actual Implementation**

_Creating Dataset_

[TFRecords for data storage with flexible format][7]
[Store data to ELWC][8]
[Create serialized data set][9]

[https://github.com/tensorflow/ranking/issues/196][10]

### Recommender system

#### Algorithm Overview

- Classifier-based
  - "Everything else" -- lots of the rest are attempts to fit the problem into a classifier model where you predict a "liked" category. So, logistic regression, etc.
- Neighborhood-based
  - User- or item-similarity-based
	- Computer similarity of users/item
	- Find k most similar users/item to User A
	- Recommend users’ items/items not seen by User A
  - Varies by choice of similarity metric
  - Varies by choice of neighborhood size
  - Possibly built on clusterings of users/items
- Latent-factor models
  - Mostly low-rank matrix factorization via ALS, SVD
  - Vary by choice of how to compute (SGD, Lanczos, etc.)
  - Also graphical models (restricted Boltzmann machines)

#### Classification Model

- Given user info, purchase history, product info, others, what’s the probability that a user will buy this product?
- Pros: Personalized, Feature can capture context, can handle limited user history
- Con: Feature might not be available, doesn’t provide as well as collaborative filtering methods

#### Co-occurrences Matrix (Collaborative Filtering)

- People who bought this also bought…
  - (\\\# items x \\\# items) matrix
	- look at diaper row to see what else people bought on the other axis
	- Recommend other items with the largest counts
  - Matrix must be normalized. Otherwise, very popular items will drown out other effects, and recommendations will be based on popularity rather than co-occurrence.
  - Similarity Measures
	- Jaccard similarity

Normalizes by popularity $
frac text{who purchased i and j} text{who purchased i or j}$

- Cosine similarity

  - The resulting similarity ranges from - 1 meaning exactly opposite, to 1 meaning the same, with 0 indicating orthogonality or decorrelation, while in-between values indicate intermediate similarity or dissimilarity.
  - Text matching. the attribute vectors A and B are usually the term frequency vectors of the documents. Cosine similarity can be seen as a method of normalizing document length during the comparison.
	- Information retrieval. the cosine similarity of two documents will range from 0 to 1 since the term frequencies (using TF–IDF weights) cannot be negative. The angle between the two term frequency vectors cannot be greater than 90°.

- Implementation
  - Given a user A bought diapers and milk, calculate personalized score of baby wipes for User A. This allows taking past purchase history into account. $S\_text{User A, wipes} = frac12(S\_text{wipes, diapers} + S\_text{wipes, milk})$
  - We can further weigh recent purchases more (time decay)
	- Sort $S\_User A, Products$ and find Products with the highest similarity.
- Drawbacks
  - Cold start:
	- A user has never purchased a product in the past.
	- A product has never been purchased with any other product.
  - Does not utilize context (time of day), user feature (age), or product feature (content-based filtering)

#### Matrix Factorization - Discover Hidden Structure

- Goal
  Use the provided rating of some users to predict the rating other users would give to these products.

- Matrix Completion Problem

Rating of movie v by user u: Users watch movies and rate them, but each user only watches a subset of movies.

- Matrix - Rating of movie v by user u.
  - $textRating(textuser, textrating)$ known for black cells
	- $textRating(textuser, textrating)$ unknown for white cells
	- Factors
	- Describe user u with topics $L\_u = [2.5, 0, 0.8, ...]$
	  - How much does she like action, romance, drama, …
	- Describe movie v with topics $R\_v = [0.3, 0.01, 1.5, ...]$
	  - How much is it action, romance, drama, …
	- See how much these two vectors agree by $
		textRating(u, v) = L\_u
		times R\_v$ for a movie. Rating will be higher for a movie that better aligns with the users’ ratings for topics.
	- If we know all users’ preferences and movie labels, combining the user preference matrix and movie matrix gives us the rating of movies by users.
	  - However, we don’t have complete info on the L and R matrices. We estimate vectors $L\_u$ and $R\_v$ so that we can eventually estimate $textrating$. Our goal is to fill in the white blocks of the rating matrix.
	  - Once we have a matrix with estimated user ratings by movies, we can identify the movies with the highest ratings by users.
- Featured matrix factorization
  - Feature-based classification approach:
	- handle cases where we might have very limited user data.
	- Time of day, what I just saw, user info, past purchases,...
  - Matrix factorization approach: capture relationships between users and items and in particular learn features of those users and those items.
	- capture groups of users who behave similarly - Women from Seattle who teach and have a baby
  - Result
	- Combine by weighing each model to mitigate the cold-start problem.
	- Ratings for a new user from features only. As more user information is discovered, matrix factorization topics become more relevant so gradually assign it with more weights.
- Performance Metrics
  - Problem with simple classification accuracy - a fraction of items correctly classified.
	- Might predict all as “not like” for better accuracy, but we are not interested in what a person does not like.
	- Limited attention span (imbalanced class problem) - recommending some items that users might not like cost more than recommending no items that users might like.
  - Precision & Recall
	- Precision-recall curve
	  - Best algorithm
	  - For a given precision, want recall as large as possible (or vice versa)
	  - One metric: largest area under the curve (AUC)
	  - Another: set desired recall and maximize precision (Precision at K)

#### Collaborative filtering - leverages the behavior of users

Build a model from a user's past behavior (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items that the user may have an interest in.

This recommender system offers feeds items that do not appear in the user's library but are often played by other users with similar interests.

Assumes that people who agreed in the past will agree in the future and that they will like similar kinds of items as they liked in the past.

- Pro
  - Does not rely on machine analyzable content - can accurately recommending complex items such as movies without requiring an "understanding" of the item itself. Cons
  - Cold start: These systems often require a large amount of existing data on a user to make accurate recommendations.
  - Scalability: There are millions of users and products. Thus, a large amount of computation power is often necessary to calculate recommendations.
  - Sparsity: The number of items sold on, say, major e-commerce sites are extremely large. The most active users will only have rated a small subset of the overall database. Thus, even the most popular items have very few ratings.
- Content-based filtering

In a content-based recommender system, keywords are used to describe the items and a user profile is built to indicate the type of item this user likes. In other words, these algorithms try to recommend items that are _similar_ to those that a user liked in the past (or is examining in the present). In particular, various candidate items are compared with items previously rated by the user, and the best-matching items are recommended. To create a user profile, the system mostly focuses on two types of information:

1. A model of the user's preference.

2. A history of the user's interaction with the recommender system. Basically, these methods use an item profile (i.e., a set of discrete attributes and features) characterizing the item within the system. The system creates a content-based profile of users based on a weighted vector of item features. The weights denote the importance of each feature to the user and can be computed from individually rated content vectors using a variety of techniques. Sophisticated methods use machine learning techniques such as Bayesian Classifiers, cluster analysis, decision trees, and artificial neural networks to estimate the probability that the user is going to like the item. Direct feedback from a user, usually in the form of a like or dislike button, can be used to assign higher or lower weights on the importance of certain attributes (using Rocchio classification or other similar techniques). Pandora uses the properties of a song or artist (a subset of the 400 attributes provided by the Music Genome Project) to seed a "station" that plays music with similar properties. User feedback is used to refine the station's results, deemphasizing certain attributes when a user "dislikes" a particular song, and emphasizing other attributes when a user "likes" a song.

- Pro
  - Pandora needs very little information to start
  - Con
	- Difficult to quantify: some content types such as clothing and movies are much harder to quantify than others such as books. It is far more limited in scope - it can only make recommendations that are similar to the original seed
	- Variety of content types: A key issue with content-based filtering is whether the system can learn user preferences from users' actions regarding one content source and use them across other content types. When the system is limited to recommending content of the same type as the user is already using, the value from the recommendation system is significantly less than when other content types from other services can be recommended. For example, recommending news articles based on browsing of news is useful, but would be much more useful when music, videos, products, discussions, etc. from different services can be recommended based on news browsing.
- Demographic and knowledge-based recommenders
- Hybrid Recommender Systems
  - Implementation

Hybrid approaches can be implemented in several ways: by making content-based and collaborative-based predictions separately and then combining them; by adding content-based capabilities to a collaborative-based approach (and vice versa); or by unifying the approaches into one model (see for a complete review of recommender systems).

- Example - Netflix

The website makes recommendations by comparing the watching and searching habits of similar users (i.e., collaborative filtering) as well as by offering movies that share characteristics with films that a user has rated highly (content-based filtering).

- Data Collection Method

  - Explicit collection
	- Asking a user to rate an item on a sliding scale.
	- Asking a user to search.
	- Asking a user to rank a collection of items from favorite to least favorite.
	- Presenting two items to a user and asking him/her to choose the better one of them.
	- Asking a user to create a list of items that he/she likes.
  - Implicit collection
	- Observing the items that a user views in an online store.
	- Analyzing item/user viewing times.
	- Keeping a record of the items that a user purchases online.
	- Obtaining a list of items that a user has listened to or watched on his/her computer.
	- Analyzing the user's social network and discovering similar likes and dislikes.

- Which system best suits Company X?

  - Weighting towards content-based because no cold start and text characteristics are easily quantifiable. As a user follows and interacts with more people in the community, increase the weight of collaborative filtering. Some goals of a recommender system
  - Diversity / Serendipity – Serendipity is a measure of "how surprising the recommendations are"
  - Recommender persistence (users may ignore items when they are shown for the first time, for instance, because they had no time to inspect the recommendations carefully)
  - Privacy – Many European countries have a strong culture of data privacy, and every attempt to introduce any level of user profiling can result in negative customer response.
  - Trust – A recommender system is of little value for a user if the user does not trust the system. Trust can be built by a recommender system by explaining how it generates recommendations, and why it recommends an item.

- Performance Measures
  - A/B testing and measure results using the following metrics
  - We approach recommendation as a ranking task, meaning that we are mainly interested in a relatively few items that we consider most relevant and are going to show to the user. This is known as top-K recommendation.
  - Ranking
	- Mean average precision: Average precision values at ranks of relevant documents. This assumes that users want to find the most relevant documents, and it is biased towards the top of the ranking. Calculation: It takes the mean of average precision (Ave.P) values across queries. Calculate the average precision of a query by looking at the ranks that have the relevant document, taking its corresponding precision, and averaging (divide by the total number of relevant documents found). Then, take the mean value of the average precision values.

```markdown
For instance, for a query for the first two documents where:
Actual items are [1, 2, 3, 4, 5]
Recommended items are [6, 4, 7, 1, 2]. AP@1 = 0
AP@2 = 0.5
MAP = (0 + 0.5)/2 = 0.25
```

- Also need to think about how getting the first document right is much more important than getting the second right and … Therefore, MAP might be biased because it assigns equal weight to all queries, while queries to the first couple items might in fact be more important. Normalized Discounted Cumulative Gain (Corrected the ranking bias in MAP)
  - Assumptions: Highly relevant documents are more useful than marginally relevant document
  - The lower the ranked position of a relevant document, the less useful it is for the user since it is less likely to be examined [Reading][11]
  - Intimidating as the name might be, the idea behind NDCG is pretty simple. A recommender returns some items and we’d like to compute how good the list is. Each item has a relevance score, usually a non-negative number. That’s gain. For items, we don’t have user feedback for we usually set the gain to zero.
  - Now we add up those scores; that’s cumulative gain. We’d prefer to see the most relevant items at the top of the list, therefore before summing the scores we divide each by a growing number (usually a logarithm of the item position) - that’s discounting - and get a DCG.
  - DCGs are not directly comparable between users, so we normalize them. The worst possible DCG when using non-negative relevance scores is zero. To get the best, we arrange all the items in the test set in the ideal order, take the first K items, and compute DCG for them. Then we divide the raw DCG by this ideal DCG to get NDCG@K, a number between 0 and 1.
  - You may have noticed that we denote the length of the recommendations list by K. It is up to the practitioner to choose this number. You can think of it as an estimate of how many items a user will have attention for, so values like 10 or 50 are common.
- Diversity
  - Intralist Similarity: the sum of pairwise similarity between two given items. Cosine similarity, Jaccard similarity coefficient.
  - Content coverage: how well the full content space is represented to the users.

E.g., [0.21 of animation, 0.1 of comedy], 0.03, 0.05, 0.56 are covered in the recommendation.

[1]:	https://github.com/airalcorn2/RankNet/blob/master/lambdarank.py
[2]:	https://papers.nips.cc/paper/2006/file/af44c4c56f385c43f2529f9b1b018f6a-Paper.pdf
[3]:	https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_tfrecord.py
[4]:	https://gist.github.com/lcamposgarrido/c28170ed8f41680a5bfdb414ab87a91f
[5]:	https://github.com/tensorflow/ranking/tree/master/tensorflow_ranking/examples/keras
[6]:	https://arxiv.org/pdf/1910.09676.pdf
[7]:	https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
[8]:	https://github.com/tensorflow/ranking/issues/196
[9]:	https://stackoverflow.com/questions/57481869/how-to-serialize-data-in-example-in-example-format-for-tensorflow-ranking
[10]:	https://github.com/tensorflow/ranking/issues/196
[11]:	https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf