##TODO:

- [x] Generate prepare small queries ( 20k )
- [ ] Generate and prepare medium queries ( 100k )
- [x] Generate and prepare all queries ( 400k )
- [ ] Save model weights, with base graph layer separate from cardinality estimation head
- [ ] Create code for experiment visualization, analysis etc. Violin plots, q-error plots with distribution, epoch development etc.
- [x] Test performance cardinality estimation on completely unseen benchmark queries
- [ ] Test impact of number of queries on unseen benchmark queries
- [ ] Find number of parameters in our model
- [ ] Use cardinality estimator for join order optimization
- [ ] Create system for easy configs for experiment running
- [ ] Generate LUBM benchmark data and test on that
- [ ] Move query generation + execution to this repository and save path, subject star, object star, complex queries separately for more fine-grained analysis.
- [ ] Move RDF2Vec to this repository
- [ ] Use pyRDF2Vec on query entities instead of on all entities in graph (for scalability), like in paper recently published
- [ ] Use the samplers from pyRDF2Vec
- [ ] Parallel random walking?

Ideas:

- Try to use pretraining with deep Q-learning
- Infer RDF2Vec embeddings when new ones arrive by taking random walks starting from this embedding and taking average of the random walk embeddings.
You could even train a model that reconstructs embeddings given random walks, so randomly _corrupt_ embedding, then give it random walks and apply model on random walks so it regenerates the embedding.
- Imitation learning from pretrained cardinality estimator. Reasoning: we copy the cardinality estimator in our weights, but as we get better at imitating we also start including more exploration.
  This exploration might help us find better join plans as we are not bound to possibly inefficient cardinality estimations
  (meaning some accuracy might be redundant, while others might be very necessary).